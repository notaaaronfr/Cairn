use async_trait::async_trait;

use crate::model::{Model, ModelError, ModelRequest, ModelResponse, ModelStream, Usage};

/// USD per million tokens. Prices change often, so Cairn never hardcodes
/// them; set them from your provider's current price sheet. A model that is
/// genuinely free should say so explicitly with `Pricing::new(0.0, 0.0)`.
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

    /// Field-wise maximum: a conservative price when the serving model of
    /// several is not known.
    pub fn max(self, other: Pricing) -> Pricing {
        Pricing {
            input: self.input.max(other.input),
            output: self.output.max(other.output),
            cache_read: self.cache_read.max(other.cache_read),
            cache_write: self.cache_write.max(other.cache_write),
        }
    }
}

/// Attaches pricing to any model, e.g. one built from a `provider:model` spec.
pub struct Priced<M> {
    inner: M,
    pricing: Pricing,
}

impl<M: Model> Priced<M> {
    pub fn new(inner: M, pricing: Pricing) -> Self {
        Self { inner, pricing }
    }
}

#[async_trait]
impl<M: Model> Model for Priced<M> {
    fn id(&self) -> String {
        self.inner.id()
    }

    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, ModelError> {
        self.inner.complete(request).await
    }

    async fn stream(&self, request: &ModelRequest) -> Result<ModelStream, ModelError> {
        self.inner.stream(request).await
    }

    fn pricing(&self) -> Option<Pricing> {
        Some(self.pricing)
    }
}
