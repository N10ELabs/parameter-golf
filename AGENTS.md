## Completion Notification

For each user request completed in this repository, send exactly one Telegram notification, and only after the task is fully complete.

Run this command as the last shell action immediately before sending the final user-facing response:

`"/Users/anthonymarti/Desktop/N10E LABS Code/scripts/telegram_notify.sh" --random`

Rules:
- Send it only after all requested work, edits, commands, and verification are complete.
- Do not send it during planning, exploration, intermediate progress, or partial completion.
- Do not send more than one notification for the same user request.
- Do not send it if the task is blocked, incomplete, failed, or waiting on the user.
- If the user explicitly asks not to receive a notification, skip it.
- If the notification command fails, mention that in the final response.
