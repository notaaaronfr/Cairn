//! AWS event-stream framing (`application/vnd.amazon.eventstream`), used by
//! Bedrock's streaming APIs. Frames carry typed headers and a payload, each
//! protected by CRC-32 checksums, which are verified here.

use std::collections::HashMap;

use cairn_runtime::ModelError;
use futures::stream::{BoxStream, StreamExt};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Frame {
    pub headers: HashMap<String, String>,
    pub payload: Vec<u8>,
}

const fn crc_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
}

static CRC: [u32; 256] = crc_table();

pub(crate) fn crc32(data: &[u8]) -> u32 {
    let mut c = !0u32;
    for &b in data {
        c = CRC[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    !c
}

fn be_u32(b: &[u8]) -> u32 {
    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
}

/// Parse one complete frame.
pub(crate) fn parse_frame(frame: &[u8]) -> Result<Frame, ModelError> {
    let bad = |m: &str| ModelError::Decode(format!("event stream: {m}"));
    if frame.len() < 16 {
        return Err(bad("frame too short"));
    }
    let total = be_u32(&frame[0..4]) as usize;
    let headers_len = be_u32(&frame[4..8]) as usize;
    if total != frame.len() || 12 + headers_len + 4 > total {
        return Err(bad("inconsistent lengths"));
    }
    if be_u32(&frame[8..12]) != crc32(&frame[0..8]) {
        return Err(bad("prelude checksum mismatch"));
    }
    if be_u32(&frame[total - 4..]) != crc32(&frame[..total - 4]) {
        return Err(bad("message checksum mismatch"));
    }
    let mut headers = HashMap::new();
    let mut h = &frame[12..12 + headers_len];
    while !h.is_empty() {
        let name_len = h[0] as usize;
        let name = String::from_utf8_lossy(h.get(1..1 + name_len).ok_or_else(|| bad("header name"))?).into_owned();
        h = &h[1 + name_len..];
        let kind = *h.first().ok_or_else(|| bad("header type"))?;
        h = &h[1..];
        let fixed =
            |n: usize| -> Result<usize, ModelError> { if h.len() >= n { Ok(n) } else { Err(bad("header value")) } };
        let (value, used) = match kind {
            0 => ("true".to_owned(), 0),
            1 => ("false".to_owned(), 0),
            2 => (h[0].to_string(), fixed(1)?),
            3 => (i16::from_be_bytes([h[0], h[1]]).to_string(), fixed(2)?),
            4 => (be_u32(h).to_string(), fixed(4)?),
            5 | 8 => (String::new(), fixed(8)?),
            6 | 7 => {
                let len =
                    u16::from_be_bytes([*h.first().ok_or_else(|| bad("len"))?, *h.get(1).ok_or_else(|| bad("len"))?])
                        as usize;
                let bytes = h.get(2..2 + len).ok_or_else(|| bad("header bytes"))?;
                (String::from_utf8_lossy(bytes).into_owned(), 2 + len)
            }
            9 => (String::new(), fixed(16)?),
            other => return Err(bad(&format!("unknown header type {other}"))),
        };
        h = &h[used..];
        headers.insert(name, value);
    }
    Ok(Frame { headers, payload: frame[12 + headers_len..total - 4].to_vec() })
}

/// Split a byte stream into frames.
pub(crate) fn frames(resp: reqwest::Response) -> BoxStream<'static, Result<Frame, ModelError>> {
    let bytes = Box::pin(resp.bytes_stream());
    Box::pin(futures::stream::unfold((bytes, Vec::<u8>::new(), false), |(mut bytes, mut buf, done)| async move {
        if done {
            return None;
        }
        loop {
            if buf.len() >= 4 {
                let total = be_u32(&buf[0..4]) as usize;
                if total < 16 {
                    return Some((
                        Err(ModelError::Decode("event stream: bad frame length".into())),
                        (bytes, buf, true),
                    ));
                }
                if buf.len() >= total {
                    let frame: Vec<u8> = buf.drain(..total).collect();
                    return Some((parse_frame(&frame), (bytes, buf, false)));
                }
            }
            match bytes.next().await {
                Some(Ok(chunk)) => buf.extend_from_slice(&chunk),
                Some(Err(e)) => return Some((Err(ModelError::Transport(e.to_string())), (bytes, buf, true))),
                None if buf.is_empty() => return None,
                None => {
                    return Some((
                        Err(ModelError::Transport("event stream ended mid-frame".into())),
                        (bytes, buf, true),
                    ));
                }
            }
        }
    }))
}

/// Encode a frame with string headers (tests and fakes).
#[cfg(test)]
pub(crate) fn encode(headers: &[(&str, &str)], payload: &[u8]) -> Vec<u8> {
    let mut h = Vec::new();
    for (k, v) in headers {
        h.push(k.len() as u8);
        h.extend_from_slice(k.as_bytes());
        h.push(7);
        h.extend_from_slice(&(v.len() as u16).to_be_bytes());
        h.extend_from_slice(v.as_bytes());
    }
    let total = 12 + h.len() + payload.len() + 4;
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&(total as u32).to_be_bytes());
    out.extend_from_slice(&(h.len() as u32).to_be_bytes());
    let prelude_crc = crc32(&out);
    out.extend_from_slice(&prelude_crc.to_be_bytes());
    out.extend_from_slice(&h);
    out.extend_from_slice(payload);
    let crc = crc32(&out);
    out.extend_from_slice(&crc.to_be_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_the_standard_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn round_trips_and_rejects_corruption() {
        let frame =
            encode(&[(":event-type", "messageStop"), (":message-type", "event")], br#"{"stopReason":"end_turn"}"#);
        let parsed = parse_frame(&frame).unwrap();
        assert_eq!(parsed.headers[":event-type"], "messageStop");
        assert_eq!(parsed.payload, br#"{"stopReason":"end_turn"}"#);

        let mut corrupt = frame.clone();
        let n = corrupt.len();
        corrupt[n - 6] ^= 1;
        assert!(parse_frame(&corrupt).unwrap_err().to_string().contains("checksum"));
    }
}
