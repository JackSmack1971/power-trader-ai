# New Feature Brainstorming Workflow

Please brainstorm a new feature or improvement based on these requirements: $ARGUMENTS

Follow this structured process to ensure the proposal is technically feasible and aligned with the project's architecture:

### Phase 1: Context Exploration
1. Use an **Explore sub-agent** to search the codebase for files related to the requested feature area.
2. Identify core architectural patterns by reading the root `CLAUDE.md` and any relevant files in `.claude/rules/`.
3. Summarize the current implementation of related features to avoid redundancy.

### Phase 2: Deep Reasoning (Ideation)
4. Use **ultrathink** mode to evaluate at least three different implementation approaches for this feature.
5. For each approach, consider:
    - **Technical Debt**: Will this require significant refactoring?
    - **Performance**: Are there potential bottlenecks?
    - **Security**: Does this introduce new attack vectors or permission requirements? 

### Phase 3: Technical Validation
6. Spawn a **Plan sub-agent** to create a high-level technical specification [12, 13]. 
7. Have the sub-agent verify that the proposed changes adhere to the coding standards defined in the project memory.

### Phase 4: Deliverables
8. Present the final brainstormed proposal including:
    - **The "WHY"**: Business logic and user value.
    - **The "WHAT"**: A concise summary of the proposed changes.
    - **The "HOW"**: A step-by-step implementation plan with file:line references to existing code.
9. Ask if I would like you to create a new Markdown file in `docs/proposals/` or a GitHub issue using the `gh` CLI for this plan
