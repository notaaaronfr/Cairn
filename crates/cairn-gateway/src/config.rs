use std::collections::BTreeMap;
use std::path::PathBuf;

use cairn_models::Pricing;
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default = "default_listen")]
    pub listen: String,
    /// JSON-lines audit log. Spend is reloaded from it on startup.
    pub audit_log: Option<PathBuf>,
    /// Allow clients to request raw `provider:model` specs, not just routes.
    #[serde(default)]
    pub allow_direct_specs: bool,
    #[serde(default)]
    pub keys: Vec<KeyConfig>,
    /// Route alias -> ordered list of `provider:model` specs (failover order).
    #[serde(default)]
    pub routes: BTreeMap<String, Vec<String>>,
    /// USD per million tokens, keyed by `provider:model` spec.
    #[serde(default)]
    pub pricing: BTreeMap<String, PriceConfig>,
}

fn default_listen() -> String {
    "127.0.0.1:8787".into()
}

#[derive(Debug, Clone, Deserialize)]
pub struct KeyConfig {
    pub name: String,
    /// Plaintext key (development only).
    pub key: Option<String>,
    /// Hex SHA-256 of the key (preferred).
    pub key_sha256: Option<String>,
    pub budget_usd: Option<f64>,
    pub budget_tokens: Option<u64>,
    /// Routes/specs this key may use. Omitted = all.
    pub models: Option<Vec<String>>,
}

impl KeyConfig {
    pub fn matches(&self, presented: &str) -> bool {
        let presented_hash = sha256_hex(presented);
        let expected = match (&self.key_sha256, &self.key) {
            (Some(h), _) => h.to_ascii_lowercase(),
            (None, Some(k)) => sha256_hex(k),
            (None, None) => return false,
        };
        // Compare digests, not raw keys, so timing does not leak key prefixes.
        presented_hash == expected
    }

    pub fn may_use(&self, model: &str) -> bool {
        self.models.as_ref().is_none_or(|m| m.iter().any(|x| x == model))
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct PriceConfig {
    pub input: f64,
    pub output: f64,
    pub cache_read: Option<f64>,
    pub cache_write: Option<f64>,
}

impl From<PriceConfig> for Pricing {
    fn from(p: PriceConfig) -> Self {
        let mut pricing = Pricing::new(p.input, p.output);
        if let Some(r) = p.cache_read {
            pricing.cache_read = r;
        }
        if let Some(w) = p.cache_write {
            pricing.cache_write = w;
        }
        pricing
    }
}

pub fn sha256_hex(s: &str) -> String {
    hex::encode(Sha256::digest(s.as_bytes()))
}

impl Config {
    pub fn from_toml(text: &str) -> Result<Self, String> {
        let config: Config = toml::from_str(text).map_err(|e| e.to_string())?;
        if config.keys.is_empty() {
            return Err("at least one [[keys]] entry is required".into());
        }
        for k in &config.keys {
            if k.key.is_none() && k.key_sha256.is_none() {
                return Err(format!("key `{}` needs `key` or `key_sha256`", k.name));
            }
            // A USD budget silently counts unpriced models as $0 and never
            // triggers, so refuse that configuration outright.
            if k.budget_usd.is_some() {
                for (route, specs) in config.routes.iter().filter(|(r, _)| k.may_use(r)) {
                    if let Some(spec) = specs.iter().find(|s| !config.pricing.contains_key(*s)) {
                        return Err(format!(
                            "key `{}` has budget_usd but route `{route}` includes `{spec}`, which has no \
                             [pricing.\"{spec}\"] entry (use input = 0.0, output = 0.0 for a free model)",
                            k.name
                        ));
                    }
                }
            }
        }
        Ok(config)
    }
}
