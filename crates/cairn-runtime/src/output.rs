//! Helpers for reading structured output out of model text.

use serde_json::Value;

/// Extract a JSON value from model text: the whole text, a fenced
/// ```json block, or the outermost `{...}` / `[...]` span.
pub fn extract_json(text: &str) -> Option<Value> {
    let t = text.trim();
    if let Ok(v) = serde_json::from_str(t) {
        return Some(v);
    }
    if let Some(start) = t.find("```") {
        let after = &t[start + 3..];
        let body = after.strip_prefix("json").or_else(|| after.strip_prefix("JSON")).unwrap_or(after);
        if let Some(end) = body.find("```") {
            if let Ok(v) = serde_json::from_str(body[..end].trim()) {
                return Some(v);
            }
        }
    }
    for (open, close) in [('{', '}'), ('[', ']')] {
        if let (Some(s), Some(e)) = (t.find(open), t.rfind(close)) {
            if s < e {
                if let Ok(v) = serde_json::from_str(&t[s..=e]) {
                    return Some(v);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extracts_json_in_common_shapes() {
        assert_eq!(extract_json(r#"{"a":1}"#), Some(json!({"a": 1})));
        assert_eq!(extract_json("Here you go:\n```json\n{\"a\": 2}\n```\nDone."), Some(json!({"a": 2})));
        assert_eq!(extract_json("Result: {\"a\": {\"b\": 3}} hope that helps"), Some(json!({"a": {"b": 3}})));
        assert_eq!(extract_json("no json here"), None);
        assert_eq!(extract_json("{\"a\": 1"), None);
    }
}
