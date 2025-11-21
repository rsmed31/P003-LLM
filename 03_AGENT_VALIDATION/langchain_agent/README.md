# langchain_agent: Orchestration Logic
Contains the agent_flow.py and all tool definitions required to sequence the E2E process, including the VPP loop logic.

# Team 3 - Agent Orchestrator

## 🤝 Contract Requirements for Team 2

### Expected Response Format

Team 2's `/v1/getAnswer` endpoint **must** return:

```json
{
  "model": "llama",
  "response": [
    {
      "device_name": "R1",
      "protocol": "OSPF",
      "configuration_mode_commands": [
        "router ospf 1",
        "network 10.0.0.0 0.0.0.255 area 0"
      ],
      "intent": [
        {
          "type": "adjacency",
          "endpoints": [
            {"role": "router", "id": "R1"},
            {"role": "router", "id": "R2"}
          ]
        }
      ]
    }
  ]
}
```

### Fallback Support

The agent also handles:
- **String format**: `{"response": "configure terminal\n..."}`
- **Direct list**: `[{"device_name": "R1", ...}]`

### Critical Fields

Each device object must contain:
- `device_name` (string): Router/switch identifier
- `configuration_mode_commands` (list of strings): CLI commands
- `protocol` (string, optional): Protocol type
- `intent` (list, optional): Validation intents

### Testing Contract Compliance

```bash
python agent_service.py --query "Configure OSPF on 3 routers" --model gemini
```

Check logs for `[T2 Parse]` messages to verify format detection.

## 📞 Team Contact for Issues

- **Invalid JSON**: Check Team 2's model output cleaning in `endpoints/inference.py`
- **Missing fields**: Verify `prompts/prompts.json` schema instructions
- **String responses**: Team 2's model may need better output format training
