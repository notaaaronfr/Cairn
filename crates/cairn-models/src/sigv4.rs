//! AWS Signature Version 4, just enough for JSON POSTs to AWS APIs.

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

type HmacSha256 = Hmac<Sha256>;

/// AWS credentials (long-term keys or temporary STS credentials).
#[derive(Clone)]
pub struct AwsCredentials {
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: Option<String>,
}

impl std::fmt::Debug for AwsCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AwsCredentials").field("access_key_id", &self.access_key_id).finish_non_exhaustive()
    }
}

impl AwsCredentials {
    /// From `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and optional `AWS_SESSION_TOKEN`.
    pub fn from_env() -> Option<Self> {
        let get = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        Some(Self {
            access_key_id: get("AWS_ACCESS_KEY_ID")?,
            secret_access_key: get("AWS_SECRET_ACCESS_KEY")?,
            session_token: get("AWS_SESSION_TOKEN"),
        })
    }
}

fn hmac(key: &[u8], data: &str) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(data.as_bytes());
    mac.finalize().into_bytes().to_vec()
}

pub(crate) fn sha256_hex(data: &[u8]) -> String {
    hex::encode(Sha256::digest(data))
}

/// RFC 3986 encoding as AWS requires: unreserved characters kept, everything
/// else percent-encoded with upper-case hex.
pub(crate) fn uri_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

pub(crate) fn signing_key(secret: &str, date: &str, region: &str, service: &str) -> Vec<u8> {
    let k_date = hmac(format!("AWS4{secret}").as_bytes(), date);
    let k_region = hmac(&k_date, region);
    let k_service = hmac(&k_region, service);
    hmac(&k_service, "aws4_request")
}

/// Canonical request and signed-header list. `headers` must already be
/// lower-case; they are sorted here.
pub(crate) fn canonical_request(
    method: &str,
    canonical_uri: &str,
    canonical_query: &str,
    headers: &[(&str, &str)],
    payload_hash: &str,
) -> (String, String) {
    let mut sorted: Vec<(&str, &str)> = headers.to_vec();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    let canonical_headers: String = sorted.iter().map(|(k, v)| format!("{k}:{}\n", v.trim())).collect();
    let signed: Vec<&str> = sorted.iter().map(|(k, _)| *k).collect();
    let signed = signed.join(";");
    (format!("{method}\n{canonical_uri}\n{canonical_query}\n{canonical_headers}\n{signed}\n{payload_hash}"), signed)
}

pub(crate) fn signature(secret: &str, amz_date: &str, region: &str, service: &str, canonical_request: &str) -> String {
    let date = &amz_date[..8];
    let scope = format!("{date}/{region}/{service}/aws4_request");
    let to_sign = format!("AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}", sha256_hex(canonical_request.as_bytes()));
    hex::encode(hmac(&signing_key(secret, date, region, service), &to_sign))
}

/// Headers to add to a signed POST: `x-amz-date`, `x-amz-content-sha256`,
/// optional `x-amz-security-token`, and `authorization`.
pub(crate) fn sign_post(
    creds: &AwsCredentials,
    region: &str,
    service: &str,
    host: &str,
    request_path: &str,
    body: &[u8],
    now: chrono::DateTime<chrono::Utc>,
) -> Vec<(&'static str, String)> {
    let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();
    let payload_hash = sha256_hex(body);
    // Non-S3 services: each path segment is URI-encoded again in the canonical form.
    let canonical_uri: String = request_path.split('/').map(uri_encode).collect::<Vec<_>>().join("/");
    let mut headers: Vec<(&str, &str)> = vec![
        ("content-type", "application/json"),
        ("host", host),
        ("x-amz-content-sha256", &payload_hash),
        ("x-amz-date", &amz_date),
    ];
    if let Some(token) = &creds.session_token {
        headers.push(("x-amz-security-token", token));
    }
    let (canonical, signed) = canonical_request("POST", &canonical_uri, "", &headers, &payload_hash);
    let sig = signature(&creds.secret_access_key, &amz_date, region, service, &canonical);
    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={}/{}/{region}/{service}/aws4_request, SignedHeaders={signed}, Signature={sig}",
        creds.access_key_id,
        &amz_date[..8]
    );
    let mut out = vec![("x-amz-date", amz_date.clone()), ("x-amz-content-sha256", payload_hash.clone())];
    if let Some(token) = &creds.session_token {
        out.push(("x-amz-security-token", token.clone()));
    }
    out.push(("authorization", authorization));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Example from the AWS Signature Version 4 documentation (IAM ListUsers).
    const SECRET: &str = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";

    #[test]
    fn derives_the_documented_signing_key() {
        let key = signing_key(SECRET, "20150830", "us-east-1", "iam");
        assert_eq!(hex::encode(key), "c4afb1cc5771d871763a393e44b703571b55cc28424d1a5e86da6ed3c154a4b9");
    }

    #[test]
    fn signs_the_documented_request() {
        let empty = sha256_hex(b"");
        let (canonical, signed) = canonical_request(
            "GET",
            "/",
            "Action=ListUsers&Version=2010-05-08",
            &[
                ("host", "iam.amazonaws.com"),
                ("content-type", "application/x-www-form-urlencoded; charset=utf-8"),
                ("x-amz-date", "20150830T123600Z"),
            ],
            &empty,
        );
        assert_eq!(signed, "content-type;host;x-amz-date");
        assert_eq!(
            sha256_hex(canonical.as_bytes()),
            "f536975d06c0309214f805bb90ccff089219ecd68b2577efef23edd43b7e1a59"
        );
        assert_eq!(
            signature(SECRET, "20150830T123600Z", "us-east-1", "iam", &canonical),
            "5d672d79c15b13162d9279b0855cfba6789a8edb4c82c400e06b5924a6f2b5d7"
        );
    }

    #[test]
    fn encodes_model_ids_in_paths() {
        assert_eq!(uri_encode("anthropic.claude-v2:1"), "anthropic.claude-v2%3A1");
        assert_eq!(uri_encode("%3A"), "%253A");
    }
}
