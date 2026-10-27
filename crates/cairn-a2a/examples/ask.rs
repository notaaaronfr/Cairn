//! Ask any A2A 1.0 agent a question, blocking and then streaming.
//!
//! ```sh
//! cargo run -p cairn-ai-a2a --example ask -- http://127.0.0.1:8791 "hello"
//! ```

use cairn_a2a::{A2aClient, Message, SendMessageRequest, SendMessageResponse, StreamResponse};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let url = args.next().unwrap_or_else(|| "http://127.0.0.1:8791".into());
    let question = args.next().unwrap_or_else(|| "hello".into());

    let client = A2aClient::discover(&url, Vec::<(String, String)>::new()).await?;
    let card = client.card().expect("discovered");
    println!("agent: {} — {}", card.name, card.description);

    match client.send_text(&question).await? {
        SendMessageResponse::Task(task) => {
            println!("task {} {:?}: {}", task.id, task.status.state, task.text());
            let again = client.get_task(&task.id, None).await?;
            println!("get_task {:?}: {}", again.status.state, again.text());
        }
        SendMessageResponse::Message(m) => println!("message: {}", m.text()),
    }

    let mut stream = client.send_streaming_message(&SendMessageRequest::new(Message::user(question))).await?;
    while let Some(item) = stream.next().await {
        match item? {
            StreamResponse::Task(t) => println!("stream task {} {:?}", t.id, t.status.state),
            StreamResponse::Message(m) => println!("stream message: {}", m.text()),
            StreamResponse::StatusUpdate(u) => println!("stream status {:?}", u.status.state),
            StreamResponse::ArtifactUpdate(a) => {
                println!("stream artifact: {}", a.artifact.parts.iter().map(|p| p.to_text()).collect::<String>())
            }
        }
    }
    Ok(())
}
