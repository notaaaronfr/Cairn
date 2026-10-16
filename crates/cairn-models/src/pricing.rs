use cairn_runtime::Usage;

/// USD per million tokens. Prices change often, so Cairn never hardcodes
/// them; set them from your provider's current price sheet.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Pricing {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

impl Pricing {
    pub fn new(input: f64, output: f64) -> Self {
        Self { input, output, cache_read: input * 0.1, cache_write: input * 1.25 }
    }

    pub fn cost(&self, u: &Usage) -> f64 {
        (u.input_tokens as f64 * self.input
            + u.output_tokens as f64 * self.output
            + u.cache_read_tokens as f64 * self.cache_read
            + u.cache_write_tokens as f64 * self.cache_write)
            / 1_000_000.0
    }
}
