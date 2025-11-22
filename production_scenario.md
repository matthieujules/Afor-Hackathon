# Production Scenario: The "Warehouse Hunt"

## 1. The Environment Construction
In a production test (or a high-fidelity simulation), we construct the environment to specifically test **Semantic Curiosity**.

*   **The "Boring" Zone**: Large open spaces with uniform textures (concrete floors, plain white walls).
    *   *Purpose:* Test if the agent correctly suppresses exploration here (low entropy, low interest).
*   **The "Breadcrumb" Trail**: A subtle trail of visual cues leading to the target.
    *   *Example:* Small oil droplets on the floor → Larger puddle → Leaking Drum (Target).
    *   *Example:* A power cable running along the floor → A server rack → A smoking power supply (Target).
*   **The "Occluded" Target**: The hazard is *not* visible from the main aisle. It is behind a stack of pallets.
    *   *Key:* The agent must see the *edge* of the pallets and decide "There is complex geometry here, I should look behind it."

## 2. The Agent's "Search" Logic (The Algorithm)

The agent runs a loop every 5 seconds:

### Step A: The Glimpse (Perception)
The robot stops and takes a high-res photo.
It sends this to **Gemini 1.5 Pro** with a structured prompt:
> "Analyze this image.
> 1. **Hazard Score (0-10):** Is there immediate danger?
> 2. **Interest Score (0-10):** Is this scene visually complex or cluttered?
> 3. **Lead Direction:** Does the clutter/interest continue off-screen? (Left/Right/Center/None)"

### Step B: The "Glow" Injection (Prediction)
*   **Scenario:** The robot sees the "Power Cable" on the floor running off to the **Left**.
*   **VLM Output:** `Interest: 8`, `Lead: Left`.
*   **Map Update:**
    *   The robot marks the cable itself as "Seen" (Green/Yellow).
    *   **CRITICAL:** It projects a cone of **High Utility (White Glow)** into the *unseen* black void to its **Left**.
    *   *Internal Monologue:* "I haven't been to the left, but the cable goes there. Therefore, the left is High Value."

### Step C: The Decision (Action)
The path planner calculates the "Utility" of all possible next moves.
*   **Move Right (Open Floor):** Utility = Low (Boring).
*   **Move Left (Into the Void):** Utility = High (High Entropy + **High Glow**).
*   **Result:** The robot turns Left, following the cable.

### Step D: The Discovery
The robot turns the corner.
*   **New View:** It sees the Smoking Power Supply.
*   **VLM Output:** `Hazard: 10`.
*   **Map Update:** The "Glow" resolves into a solid "Red" Hazard zone.
*   **Action:** Stop and alert human.

## Why this is "Production Ready"
This mimics human intuition. We don't scan every inch of a wall. We scan until we see a *feature*, then we fixate on that feature and follow it until it resolves. This minimizes the number of frames needed to find the target (Efficiency).
