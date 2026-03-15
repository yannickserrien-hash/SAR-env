# Visualization & Web Interface

## Overview

The system uses Flask (port 3000) serving a JavaScript-based web interface for real-time visualization of the multi-agent environment. The frontend polls a separate REST API (port 3001) for state updates and renders agent perspectives, god view, and handles human control inputs.

## Architecture

**Backend**: Flask server at `SaR_gui/visualization_server.py` serves static HTML/JS/CSS. The MATRX API at `matrx/api/api.py` (port 3001) provides state data via RESTful endpoints.

**Frontend**: Polling-based architecture (no websockets). JavaScript in `SaR_gui/static/js/loop.js` requests updates every ~300-500ms using `requestAnimationFrame`, receives world state, and triggers redraws.

**Views**: Three distinct perspectives served by Flask routes:
- `/god` - Full world state with all agents visible
- `/agent/<id>` - Read-only agent perspective (filtered state)
- `/human-agent/<id>` - Interactive agent view with keyboard/chat input

## State Streaming Pattern

The visualization loop (`loop.js`) continuously:
1. Polls `/get_latest_state_and_messages` with agent ID and chat offsets
2. Receives filtered state dict containing only what that agent can see
3. Extracts tick number, world settings, messages from response
4. Calls `draw()` in `gen_grid.js` to render objects on grid

State filtering happens server-side in the API. Each agent view receives `data['states'][tick][agent_id]['state']` containing only visible objects and world metadata. The god view gets complete state with all agents.

## Real-Time Rendering

`gen_grid.js` handles all rendering logic. Objects are positioned on a CSS grid where each tile is dynamically sized based on world dimensions. Movement animations use CSS transitions calculated from tick duration. Objects track previous/current positions to enable smooth movement between ticks.

Score display updates via chat messages. When a message contains "Our score is", `toolbar.js` extracts the score value and updates the `#score` element in the toolbar. Elapsed time is not explicitly tracked in current code.

## Human Control Input

For `/human-agent/<id>` views, `human_agent.js` binds keyboard listeners that capture arrow keys and send them to `/send_userinput/<agent_id>`. The API stores userinput in `_userinput` dict, which agent brains read during their decision cycle.

The human agent template includes extensive button-based chat controls for rescue operations, victim reporting, and area searches - all domain-specific to the SAR scenario.

## Key Files

- `SaR_gui/visualization_server.py` - Flask routes and server startup
- `SaR_gui/static/js/loop.js` - Polling loop and state fetching
- `SaR_gui/static/js/gen_grid.js` - Grid rendering and object visualization
- `SaR_gui/static/js/toolbar.js` - Chat, score display, play/pause controls
- `SaR_gui/static/js/human_agent.js` - Keyboard input routing
- `SaR_gui/templates/*.html` - View templates (god, agent, human-agent, start)
- `matrx/api/api.py` - REST API providing state data (port 3001)

## Communication Flow

```
GridWorld (Python)
  → API stores state in __states dict
  → Frontend polls /get_latest_state_and_messages
  → API returns filtered state for agent_id
  → loop.js receives JSON, calls draw()
  → gen_grid.js renders to DOM
  → human_agent.js sends input back to API
  → API stores in _userinput
  → Agent brain reads userinput on next tick
```

No websockets or server push. The frontend aggressively polls (configurable via tick duration) and the API returns cached state. This simple architecture scales to multiple simultaneous viewers.
