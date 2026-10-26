//! Minimal server-sent-events reader and a helper that turns provider event
//! streams into Cairn [`StreamEvent`]s.

use std::collections::VecDeque;

use cairn_runtime::{ModelError, ModelStream, StreamEvent};
use futures::stream::{BoxStream, Stream, StreamExt};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

fn parse_block(block: &str) -> Option<SseEvent> {
    let mut event = None;
    let mut data = Vec::new();
    for line in block.lines() {
        if let Some(v) = line.strip_prefix("event:") {
            event = Some(v.trim().to_owned());
        } else if let Some(v) = line.strip_prefix("data:") {
            data.push(v.strip_prefix(' ').unwrap_or(v).to_owned());
        }
    }
    (!data.is_empty()).then(|| SseEvent { event, data: data.join("\n") })
}

#[cfg(test)]
pub(crate) fn parse_text(text: &str) -> Vec<SseEvent> {
    text.replace('\r', "").split("\n\n").filter_map(parse_block).collect()
}

/// Read SSE events from an HTTP response body.
pub(crate) fn events(resp: reqwest::Response) -> BoxStream<'static, Result<SseEvent, ModelError>> {
    let bytes = Box::pin(resp.bytes_stream());
    Box::pin(futures::stream::unfold((bytes, Vec::<u8>::new(), false), |(mut bytes, mut buf, done)| async move {
        if done {
            return None;
        }
        loop {
            if let Some(pos) = buf.windows(2).position(|w| w == b"\n\n") {
                let block: Vec<u8> = buf.drain(..pos + 2).collect();
                if let Some(ev) = parse_block(&String::from_utf8_lossy(&block)) {
                    return Some((Ok(ev), (bytes, buf, false)));
                }
                continue;
            }
            match bytes.next().await {
                Some(Ok(chunk)) => buf.extend(chunk.iter().filter(|b| **b != b'\r')),
                Some(Err(e)) => return Some((Err(ModelError::Transport(e.to_string())), (bytes, buf, true))),
                None => {
                    let rest = parse_block(&String::from_utf8_lossy(&buf));
                    return rest.map(|ev| (Ok(ev), (bytes, Vec::new(), true)));
                }
            }
        }
    }))
}

/// Drive a provider-specific accumulator over SSE events.
///
/// `step` maps one SSE event to zero or more stream events; `end` runs once
/// when the transport closes (used to emit `Completed` if the provider did
/// not signal it explicitly).
pub(crate) fn translate<A, S>(
    sse: S,
    acc: A,
    step: fn(&mut A, SseEvent) -> Result<Vec<StreamEvent>, ModelError>,
    end: fn(&mut A) -> Result<Vec<StreamEvent>, ModelError>,
) -> ModelStream
where
    A: Send + 'static,
    S: Stream<Item = Result<SseEvent, ModelError>> + Send + 'static,
{
    let sse = Box::pin(sse);
    let state = (sse, acc, VecDeque::<StreamEvent>::new(), false, false);
    Box::pin(futures::stream::unfold(
        state,
        move |(mut sse, mut acc, mut queue, mut ended, mut completed)| async move {
            loop {
                if let Some(ev) = queue.pop_front() {
                    if matches!(ev, StreamEvent::Completed(_)) {
                        completed = true;
                    }
                    return Some((Ok(ev), (sse, acc, queue, ended, completed)));
                }
                if ended || completed {
                    return None;
                }
                match sse.next().await {
                    Some(Ok(raw)) => match step(&mut acc, raw) {
                        Ok(evs) => queue.extend(evs),
                        Err(e) => return Some((Err(e), (sse, acc, queue, true, completed))),
                    },
                    Some(Err(e)) => return Some((Err(e), (sse, acc, queue, true, completed))),
                    None => {
                        ended = true;
                        match end(&mut acc) {
                            Ok(evs) => queue.extend(evs),
                            Err(e) => return Some((Err(e), (sse, acc, queue, true, completed))),
                        }
                    }
                }
            }
        },
    ))
}

/// Error for a non-2xx response to a streaming request.
pub(crate) async fn check_status(resp: reqwest::Response) -> Result<reqwest::Response, ModelError> {
    let status = resp.status();
    if status.is_success() {
        return Ok(resp);
    }
    let body = resp.text().await.unwrap_or_default();
    Err(ModelError::Api { status: status.as_u16(), body })
}
