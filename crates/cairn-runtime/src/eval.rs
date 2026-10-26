//! Evals by replay: re-run recorded runs on a candidate configuration (e.g.
//! another model), score the results and compare cost.
//!
//! Replays never repeat real side effects. A tool call that matches one in
//! the recorded run gets the recorded result. An unmatched `Read` call may
//! run live ([`LiveTools::ReadOnly`], the default); unmatched `Write` and
//! `Irreversible` calls are never executed and the model is told so.
//! Replays use their own in-memory journal, never the production one.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use serde::Serialize;
use serde_json::Value;

use crate::agent::{Agent, RunOutcome};
use crate::event::{Event, EventKind};
use crate::journal::{InMemoryJournal, Journal, JournalError};
use crate::message::{Content, Message};
use crate::model::{Model, Usage};
use crate::policy::AllowAll;
use crate::state::{RunState, Status};
use crate::tool::{Effect, Tool, ToolCtx, ToolError, ToolSpec};

/// What a run achieved: the recorded baseline or a replayed candidate.
#[derive(Debug, Clone, Serialize)]
pub struct RunSummary {
    pub status: Status,
    pub output: Option<String>,
    pub structured: Option<Value>,
    pub steps: u32,
    pub usage: Usage,
    pub cost_usd: f64,
}

impl RunSummary {
    fn from_state(s: &RunState) -> Self {
        Self {
            output: match &s.status {
                Status::Completed { output } => Some(output.clone()),
                _ => None,
            },
            status: s.status.clone(),
            structured: s.structured_output.clone(),
            steps: s.steps,
            usage: s.usage,
            cost_usd: s.cost_usd,
        }
    }

    fn from_outcome(o: &RunOutcome) -> Self {
        Self {
            status: o.status.clone(),
            output: o.output().map(str::to_owned),
            structured: o.structured.clone(),
            steps: o.steps,
            usage: o.usage,
            cost_usd: o.cost_usd,
        }
    }

    pub fn completed(&self) -> bool {
        matches!(self.status, Status::Completed { .. })
    }
}

#[derive(Debug, Clone)]
struct Recording {
    tool: String,
    input: Value,
    output: Value,
    is_error: bool,
}

/// One recorded run to replay.
#[derive(Debug, Clone)]
pub struct EvalCase {
    pub run_id: String,
    pub input: Message,
    pub baseline: RunSummary,
    recordings: Vec<Recording>,
}

impl EvalCase {
    /// Build a case from a run's journal. `None` if the run never started.
    pub fn from_events(events: &[Event]) -> Option<Self> {
        let state = RunState::from_events(events)?;
        let input = events.iter().find_map(|e| match &e.kind {
            EventKind::RunStarted { input, .. } => Some(input.clone()),
            _ => None,
        })?;
        let mut calls: HashMap<String, (String, Value)> = HashMap::new();
        let mut recordings = Vec::new();
        for e in events {
            match &e.kind {
                EventKind::ModelCalled { response, .. } => {
                    for c in &response.content {
                        if let Content::ToolUse { id, name, input } = c {
                            calls.insert(id.clone(), (name.clone(), input.clone()));
                        }
                    }
                }
                EventKind::ToolCompleted { call_id, output, is_error, .. } => {
                    if let Some((tool, input)) = calls.get(call_id) {
                        recordings.push(Recording {
                            tool: tool.clone(),
                            input: input.clone(),
                            output: output.clone(),
                            is_error: *is_error,
                        });
                    }
                }
                _ => {}
            }
        }
        Some(Self { run_id: state.run_id.clone(), input, baseline: RunSummary::from_state(&state), recordings })
    }

    /// Load cases for these run ids from a journal (skipping unknown ids).
    pub async fn load(journal: &dyn Journal, run_ids: &[String]) -> Result<Vec<Self>, JournalError> {
        let mut cases = Vec::new();
        for id in run_ids {
            if let Some(case) = Self::from_events(&journal.load(id).await?) {
                cases.push(case);
            }
        }
        Ok(cases)
    }
}

/// Whether unmatched tool calls may execute during a replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LiveTools {
    /// Only recorded results are served; every unmatched call errors.
    Never,
    /// Unmatched `Read` tools run live; writes never do.
    #[default]
    ReadOnly,
}

/// Serves recorded results for a tool during a replay.
struct ReplayTool {
    inner: Arc<dyn Tool>,
    recorded: Mutex<VecDeque<Recording>>,
    live: LiveTools,
}

#[async_trait]
impl Tool for ReplayTool {
    fn spec(&self) -> &ToolSpec {
        self.inner.spec()
    }

    fn timeout(&self) -> Option<std::time::Duration> {
        self.inner.timeout()
    }

    async fn call(&self, input: Value, ctx: &ToolCtx) -> Result<Value, ToolError> {
        let hit = {
            let mut recorded = self.recorded.lock().unwrap();
            recorded.iter().position(|r| r.input == input).and_then(|i| recorded.remove(i))
        };
        if let Some(r) = hit {
            return if r.is_error {
                Err(ToolError::new(
                    r.output["error"].as_str().map(str::to_owned).unwrap_or_else(|| r.output.to_string()),
                ))
            } else {
                Ok(r.output)
            };
        }
        let spec = self.inner.spec();
        if self.live == LiveTools::ReadOnly && spec.effect == Effect::Read {
            return self.inner.call(input, ctx).await;
        }
        Err(ToolError::new(format!(
            "`{}` was not executed: this is an evaluation replay and no recorded result matches these arguments",
            spec.name
        )))
    }
}

/// A 0–1 score with an explanation.
#[derive(Debug, Clone, Serialize)]
pub struct Score {
    pub value: f64,
    pub note: String,
    /// What scoring itself cost (e.g. an LLM judge).
    pub cost_usd: f64,
}

impl Score {
    pub fn new(value: f64, note: impl Into<String>) -> Self {
        Self { value: value.clamp(0.0, 1.0), note: note.into(), cost_usd: 0.0 }
    }
}

#[async_trait]
pub trait Scorer: Send + Sync {
    fn name(&self) -> String;
    async fn score(&self, case: &EvalCase, candidate: &RunSummary) -> Score;
}

/// 1 if the candidate run completed (including any finish schema).
pub struct Completed;

#[async_trait]
impl Scorer for Completed {
    fn name(&self) -> String {
        "completed".into()
    }
    async fn score(&self, _: &EvalCase, c: &RunSummary) -> Score {
        if c.completed() { Score::new(1.0, "completed") } else { Score::new(0.0, format!("{:?}", c.status)) }
    }
}

/// Fraction of these structured-output fields equal to the baseline's.
pub struct FieldsMatch(pub Vec<String>);

impl FieldsMatch {
    pub fn new(fields: &[&str]) -> Self {
        Self(fields.iter().map(|f| (*f).to_owned()).collect())
    }
}

#[async_trait]
impl Scorer for FieldsMatch {
    fn name(&self) -> String {
        format!("fields_match({})", self.0.join(","))
    }
    async fn score(&self, case: &EvalCase, c: &RunSummary) -> Score {
        let (Some(b), Some(s)) = (&case.baseline.structured, &c.structured) else {
            return Score::new(0.0, "missing structured output");
        };
        let differing: Vec<&str> =
            self.0.iter().filter(|f| b[f.as_str()] != s[f.as_str()]).map(String::as_str).collect();
        let value = 1.0 - differing.len() as f64 / self.0.len().max(1) as f64;
        Score::new(
            value,
            if differing.is_empty() { "all match".into() } else { format!("differs: {}", differing.join(", ")) },
        )
    }
}

/// An LLM grades the candidate's answer against the baseline's.
pub struct Judge {
    model: Arc<dyn Model>,
    rubric: String,
}

impl Judge {
    pub fn new(model: impl Model + 'static, rubric: impl Into<String>) -> Self {
        Self { model: Arc::new(model), rubric: rubric.into() }
    }
}

#[async_trait]
impl Scorer for Judge {
    fn name(&self) -> String {
        "judge".into()
    }

    async fn score(&self, case: &EvalCase, c: &RunSummary) -> Score {
        let answer = |s: &RunSummary| {
            s.structured
                .as_ref()
                .map(Value::to_string)
                .or_else(|| s.output.clone())
                .unwrap_or_else(|| "(no answer)".into())
        };
        let prompt = format!(
            "Grade a candidate answer against a reference answer to the same task.\n\nRubric: {}\n\n\
             Task:\n{}\n\nReference answer:\n{}\n\nCandidate answer:\n{}\n\n\
             Reply with only JSON: {{\"score\": <0.0 to 1.0>, \"reason\": \"<one sentence>\"}}",
            self.rubric,
            case.input.text(),
            answer(&case.baseline),
            answer(c),
        );
        let request =
            crate::model::ModelRequest { messages: vec![Message::user(prompt)], max_tokens: 512, ..Default::default() };
        match self.model.complete(&request).await {
            Ok(resp) => {
                let cost = self.model.cost_usd(&resp).unwrap_or(0.0);
                let text = crate::message::content_text(&resp.content);
                let parsed = crate::output::extract_json(&text);
                let mut score = match parsed.as_ref().and_then(|v| v["score"].as_f64()) {
                    Some(v) => Score::new(v, parsed.as_ref().and_then(|p| p["reason"].as_str()).unwrap_or_default()),
                    None => Score::new(0.0, format!("judge reply unreadable: {text}")),
                };
                score.cost_usd = cost;
                score
            }
            Err(e) => Score::new(0.0, format!("judge failed: {e}")),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CaseResult {
    pub run_id: String,
    pub baseline: RunSummary,
    pub candidate: Option<RunSummary>,
    pub scores: Vec<(String, Score)>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvalReport {
    pub cases: Vec<CaseResult>,
}

impl EvalReport {
    /// Mean of one scorer across cases (cases that errored count as 0).
    pub fn mean(&self, scorer: &str) -> f64 {
        if self.cases.is_empty() {
            return 0.0;
        }
        let total: f64 =
            self.cases.iter().map(|c| c.scores.iter().find(|(n, _)| n == scorer).map_or(0.0, |(_, s)| s.value)).sum();
        total / self.cases.len() as f64
    }

    /// Share of cases where every score is at least `threshold`.
    pub fn pass_rate(&self, threshold: f64) -> f64 {
        if self.cases.is_empty() {
            return 0.0;
        }
        let passed = self
            .cases
            .iter()
            .filter(|c| c.error.is_none() && !c.scores.is_empty() && c.scores.iter().all(|(_, s)| s.value >= threshold))
            .count();
        passed as f64 / self.cases.len() as f64
    }

    pub fn baseline_cost(&self) -> f64 {
        self.cases.iter().map(|c| c.baseline.cost_usd).sum()
    }

    /// Candidate run cost plus scoring cost.
    pub fn candidate_cost(&self) -> f64 {
        self.cases
            .iter()
            .map(|c| {
                c.candidate.as_ref().map_or(0.0, |r| r.cost_usd) + c.scores.iter().map(|(_, s)| s.cost_usd).sum::<f64>()
            })
            .sum()
    }

    pub fn to_markdown(&self) -> String {
        let names: Vec<String> =
            self.cases.first().map(|c| c.scores.iter().map(|(n, _)| n.clone()).collect()).unwrap_or_default();
        let mut out = format!("| run | baseline $ | candidate $ | steps (b→c) | {} |\n", names.join(" | "));
        out += &format!("|---|---|---|---|{}\n", "---|".repeat(names.len()));
        for c in &self.cases {
            let cand = c.candidate.as_ref();
            let scores: Vec<String> = c.scores.iter().map(|(_, s)| format!("{:.2}", s.value)).collect();
            out += &format!(
                "| {} | {:.6} | {} | {}→{} | {} |\n",
                &c.run_id[..c.run_id.len().min(13)],
                c.baseline.cost_usd,
                cand.map_or("error".into(), |r| format!("{:.6}", r.cost_usd)),
                c.baseline.steps,
                cand.map_or("-".into(), |r| r.steps.to_string()),
                if c.error.is_some() { c.error.clone().unwrap_or_default() } else { scores.join(" | ") }
            );
        }
        let means: Vec<String> = names.iter().map(|n| format!("{n} {:.2}", self.mean(n))).collect();
        out += &format!(
            "\n**Mean:** {}  ·  **Cost:** baseline ${:.6} vs candidate ${:.6}\n",
            means.join(", "),
            self.baseline_cost(),
            self.candidate_cost()
        );
        out
    }
}

/// Replays cases against a candidate configuration derived from an agent.
pub struct Evaluator {
    agent: Agent,
    scorers: Vec<Arc<dyn Scorer>>,
    live: LiveTools,
    concurrency: usize,
}

impl Evaluator {
    /// `candidate` is the configuration to test, typically
    /// `production.to_builder().model(k2).build()?`.
    pub fn new(candidate: Agent) -> Self {
        Self { agent: candidate, scorers: Vec::new(), live: LiveTools::default(), concurrency: 4 }
    }

    pub fn scorer(mut self, s: impl Scorer + 'static) -> Self {
        self.scorers.push(Arc::new(s));
        self
    }

    pub fn live_tools(mut self, live: LiveTools) -> Self {
        self.live = live;
        self
    }

    pub fn concurrency(mut self, n: usize) -> Self {
        self.concurrency = n.max(1);
        self
    }

    pub async fn run(&self, cases: &[EvalCase]) -> EvalReport {
        let results = futures::stream::iter(cases.iter().map(|case| self.run_case(case)))
            .buffered(self.concurrency)
            .collect::<Vec<_>>()
            .await;
        EvalReport { cases: results }
    }

    async fn run_case(&self, case: &EvalCase) -> CaseResult {
        let builder = self.agent.to_builder();
        let tools: Vec<Arc<dyn Tool>> = builder
            .tool_list()
            .iter()
            .map(|t| {
                let recorded = case.recordings.iter().filter(|r| r.tool == t.spec().name).cloned().collect();
                Arc::new(ReplayTool { inner: t.clone(), recorded: Mutex::new(recorded), live: self.live })
                    as Arc<dyn Tool>
            })
            .collect();
        let agent = builder
            .replace_tools(tools)
            .journal(InMemoryJournal::new())
            // Writes are already neutralised by ReplayTool, so approvals would only stall the replay.
            .policy(AllowAll)
            .without_on_delta()
            .build();
        let result = match agent {
            Ok(agent) => agent.run_message(case.input.clone()).await.map_err(|e| e.to_string()),
            Err(e) => Err(e.to_string()),
        };
        match result {
            Ok(outcome) => {
                let candidate = RunSummary::from_outcome(&outcome);
                let mut scores = Vec::new();
                for s in &self.scorers {
                    scores.push((s.name(), s.score(case, &candidate).await));
                }
                CaseResult {
                    run_id: case.run_id.clone(),
                    baseline: case.baseline.clone(),
                    candidate: Some(candidate),
                    scores,
                    error: None,
                }
            }
            Err(error) => CaseResult {
                run_id: case.run_id.clone(),
                baseline: case.baseline.clone(),
                candidate: None,
                scores: Vec::new(),
                error: Some(error),
            },
        }
    }
}
