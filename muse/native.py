"""Adapter for unchanged V5 HookClient PostToolUse/PostToolUseFailure events."""

from .context import current_assistant, ALLOWED


def make_step_callback(runtime, assistant_provider=None, record_transform=None):
    async def step(client, event):
        hook = event["hook"]
        if hook.get("tool_name") not in ALLOWED:
            return
        if assistant_provider is None:
            assistant, turn_id = current_assistant(client, event)
        else:
            assistant, turn_id = assistant_provider(client, event)
        hook = event["hook"]
        if record_transform is not None:
            assistant = record_transform(assistant)
            hook = record_transform(hook)
        if (
            runtime.conf.get("gate_enabled", True)
            and hasattr(client, "rpc")
            and hasattr(client, "session")
        ):
            snapshot = await client.rpc(
                "session/messages", {"sessionId": client.session}
            )
            runtime.public_message_snapshot = (
                record_transform(snapshot["messages"])
                if record_transform is not None
                else snapshot["messages"]
            )
        try:
            await runtime.on_checkpoint(assistant, hook, turn_id, client.guide)
        finally:
            runtime.public_message_snapshot = None

    return step
