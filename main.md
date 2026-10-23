You are an expert Python developer, operations research engineer, and machine learning engineer. Help me build a complete, working software project for HackSHSID 2026 Track 2: AI Data Center Power–Computing Co-Scheduling.

IMPORTANT: Do not just explain the solution or generate a static UI. Actually implement the code, run it, test it, debug it, and make sure the entire project works.

I am a high school student participating in a STEM competition. The project should be technically impressive but understandable enough for me to explain the algorithms, assumptions, mathematical models, and experimental results to judges.

# 1. Problem Definition

Build an intelligent scheduling system for an AI data center.

The data center faces three changing conditions:

1. Electricity prices change over time.
2. Renewable electricity generation, such as solar and wind power, is variable.
3. Computing workloads arrive over time, and different tasks have different deadlines and priorities.

The system must decide:

- When computing tasks should execute.
- Which tasks should be delayed and which must execute immediately.
- How much computing capacity should be used in each time period.
- How renewable electricity and grid electricity should be allocated.
- Whether a battery should charge or discharge, if battery storage is included in the model.

The objective is to reduce electricity costs and carbon emissions without violating service requirements.

There is no provided competition dataset. Create a configurable synthetic dataset, clearly document all assumptions, and make it possible to replace the synthetic data with real data later.

# 2. Technology Stack

Use the following technologies unless there is a compelling technical reason not to:

- Python 3.11 or later
- NumPy and pandas for data processing
- Matplotlib for charts
- Streamlit for the interactive dashboard
- SciPy optimization tools or CVXPY for mathematical optimization
- pytest for testing

Prefer a transparent mathematical optimization model over an unnecessarily complicated black-box machine learning model.

Keep the architecture modular, readable, and easy to maintain.

# 3. Synthetic Data Generation

Create a realistic synthetic dataset covering at least 24 hours, divided into 15-minute time intervals.

Support multiple scenarios and random seeds.

Generate the following variables for every time interval:

A. Electricity:

- Grid electricity price.
- Grid electricity carbon intensity.
- Available renewable electricity.
- Optional renewable electricity curtailment.

B. Data center:

- Maximum computing capacity.
- Computing power consumption.
- Baseline workload demand.
- Remaining computing capacity.

C. Computing tasks:
Each task should have:

- Task ID.
- Arrival time.
- Required computing workload.
- Deadline.
- Priority.
- Whether the task is delayable.
- Whether the task is interruptible.
- Optional penalty for missing its deadline.

Use at least three workload categories:

1. Urgent inference tasks:
   - Must be processed promptly.
   - Have strict latency or deadline requirements.
   - Cannot be arbitrarily delayed.
2. Flexible batch inference tasks:
   - Can be scheduled within a permitted time window.
   - Have moderate deadlines.
3. Model training tasks:
   - Can be delayed or paused within reasonable limits.
   - Should preferentially execute when electricity is cheap or renewable energy is abundant.

Make the synthetic data reasonably realistic.

For example:

- Solar generation should be higher during daylight.
- Electricity prices should fluctuate.
- Computing demand should vary throughout the day.
- Renewable generation and computing demand should not always peak simultaneously.

Avoid making the data so simple that the optimization algorithm always produces the same obvious schedule.

Save generated datasets as CSV files so that experiments can be reproduced.

# 4. Mathematical Optimization Model

Implement a genuine scheduling optimizer.

Define the decision variables, parameters, objective function, and constraints explicitly in the code and documentation.

At a minimum, include:

A. Decision variables:

- Computing workload scheduled in each time interval.
- Grid electricity consumption.
- Renewable electricity utilization.
- Optionally, battery charging and discharging.

B. Objective function:

Minimize a weighted total cost consisting of:

1. Electricity expenditure.
2. Carbon emissions or carbon-related penalties.
3. Penalties for deadline violations.
4. Penalties for failing to serve urgent demand.
5. Optional penalties for unnecessary workload switching.

Use configurable weights for cost and emissions.

Do not arbitrarily choose weights without explaining their purpose. Make the weights configurable through the dashboard.

C. Constraints:

- Computing capacity cannot exceed the data center's available capacity.
- Electricity consumption must be sufficient to support scheduled computing.
- Grid electricity consumption cannot be negative.
- Renewable energy usage cannot exceed renewable energy availability.
- Each task must be scheduled within its permitted execution window, unless the model explicitly represents a deadline violation.
- Workloads must not be processed before their arrival.
- Non-interruptible tasks cannot be divided across incompatible execution periods.
- Urgent workloads must satisfy their service requirements whenever feasible.
- If a battery is implemented, its state of charge must remain within its limits, and charging/discharging efficiencies must be modeled correctly.

Use consistent units throughout the system. Document the conversion between workload units, computing capacity, power consumption, and energy consumption.

Do not introduce constraints that are mathematically inconsistent or make the model infeasible under ordinary scenarios.

If a task cannot meet its deadline because of insufficient capacity, report the infeasibility or deadline violation rather than silently dropping the task.

# 5. Optimization Algorithm

Implement at least one real optimization algorithm.

Preferred approach:

- Use a linear programming or mixed-integer linear programming formulation where appropriate.
- Use SciPy or CVXPY for the optimization.
- Use binary decision variables only when necessary.
- Keep the initial model computationally manageable.

The optimizer should make scheduling decisions across the entire planning horizon, rather than simply assigning all workloads to the cheapest time interval.

If a full optimization model becomes too complicated, first implement a correct and functioning simplified model, then extend it.

Also implement a simple baseline scheduling method for comparison.

The baseline should represent a reasonable data center without intelligent energy-aware scheduling. For example, it could process workloads as early as possible while respecting deadlines and capacity constraints.

Ensure the baseline and proposed method process equivalent workloads and use identical electricity-price, renewable-generation, and capacity data.

Do not create an unfair baseline that makes the proposed algorithm look artificially better.

# 6. Optional Rolling-Horizon Scheduling

After the main optimization model works, implement an optional rolling-horizon mode.

The system should:

1. Observe the current time interval and available information.
2. Optimize the next planning horizon.
3. Execute only the first scheduling decision.
4. Advance to the next time interval.
5. Update electricity prices, renewable generation, and arriving workloads.
6. Re-optimize.

Compare this approach with a full-day schedule that knows the entire day's conditions in advance.

Clearly distinguish perfect-forecast optimization from realistic online scheduling.

If future electricity prices or renewable generation are unknown in rolling-horizon mode, use forecasts or scenarios rather than allowing the optimizer to access future actual values.

# 7. Experimental Evaluation

The competition explicitly requires a comparison with a simple unscheduled or non-energy-aware approach, an explanation of why the proposed method is credible, and an assessment of when it might fail.

Implement an experimental framework that evaluates both the baseline and proposed algorithm on identical scenarios.

Calculate at least the following metrics:

A. Economic performance:

- Total electricity cost.
- Cost reduction percentage compared with baseline.
- Average electricity cost per unit of completed workload.

B. Environmental performance:

- Total operational carbon emissions.
- Emissions reduction percentage.
- Renewable energy utilization rate.
- Percentage of total electricity consumption supplied by renewable energy.

C. Service performance:

- Total completed workload.
- Number of missed deadlines.
- Deadline miss rate.
- Average task completion delay.
- Urgent task service rate.

D. Computational performance:

- Optimization runtime.
- Solver status.
- Number of infeasible or failed scenarios.

Use mathematically correct metric definitions.

For example:

Cost reduction (%) =
(Baseline cost - Optimized cost) / Baseline cost × 100%.

Emissions reduction (%) =
(Baseline emissions - Optimized emissions) / Baseline emissions × 100%.

Handle zero denominators safely.

Do not fabricate performance improvements. All reported results must be calculated from actual executions of the implemented algorithms.

# 8. Stress Testing and Robustness

Implement multiple test scenarios.

At minimum, include:

1. Normal conditions:
   Moderate workload and ordinary renewable generation.
2. Renewable energy shortage:
   Renewable electricity availability falls substantially.
3. Electricity price spike:
   Grid electricity becomes temporarily very expensive.
4. High demand:
   Computing demand approaches or exceeds available capacity.
5. Sudden workload surge:
   Many tasks arrive unexpectedly.
6. Forecast error:
   Predicted prices or renewable generation differ from actual values.
7. Carbon-priority scenario:
   The optimizer places greater importance on reducing emissions.
8. Cost-priority scenario:
   The optimizer places greater importance on minimizing electricity expenditure.

For each scenario, report the baseline and optimized results in a comparison table.

Analyze situations where the optimizer performs poorly or cannot meet all deadlines.

If possible, perform repeated experiments with different random seeds and report mean performance and variability.

# 9. Optimality and Algorithm Validation

Do not claim that the optimizer is globally optimal unless the mathematical formulation and solver status justify that claim.

For small test cases:

- Compare the optimizer against an exhaustive search or another suitable exact method when feasible.
- Verify that the reported objective value satisfies the model's constraints.
- Check that all energy and workload balances are consistent.

For larger test cases:

- Report the solver status.
- Report the optimality gap if the selected solver provides it.
- Report runtime and any convergence limitations.

Include automated tests for:

- Workload conservation.
- Energy balance.
- Computing capacity.
- Task arrival and deadline constraints.
- Renewable energy availability.
- Battery state-of-charge constraints, if applicable.
- Correct calculation of costs and emissions.
- Reproducibility using fixed random seeds.

# 10. Interactive Streamlit Dashboard

Build a polished, professional dashboard suitable for a technology competition presentation.

The dashboard should include the following pages or sections.

A. Overview:

- Project title.
- Short explanation of the scheduling problem.
- Key performance indicators.
- Cost savings and emissions reductions.
- Clear comparison between baseline and optimized scheduling.

B. Energy and Demand:

- Electricity-price curve over time.
- Renewable-generation curve.
- Computing-demand curve.
- Combined chart showing when electricity is cheap, green energy is abundant, and computing demand is high.

C. Scheduling Results:

- A time-series chart of scheduled computing workload.
- Grid electricity consumption.
- Renewable energy utilization.
- Battery behavior, if implemented.
- A timeline or Gantt chart of individual task execution windows.

D. Algorithm Comparison:

- Side-by-side comparison of baseline and optimized metrics.
- Cost comparison chart.
- Emissions comparison chart.
- Deadline performance comparison.
- Runtime comparison.

E. Scenario Testing:

- Scenario selector.
- Adjustable electricity-price and renewable-generation parameters.
- Workload intensity controls.
- Cost-versus-carbon weight controls.
- Button to run the optimization again.
- Results for each scenario.

F. Methodology:

- Mathematical objective function.
- Definitions of the decision variables.
- Explanation of constraints.
- Algorithm description.
- Model assumptions and limitations.

G. Export:

- Download the generated dataset as CSV.
- Download task schedules as CSV.
- Download experiment results as CSV.
- Export a summary of the current experiment.

All dashboard metrics and charts must be connected to actual model outputs. Do not use hardcoded demonstration values.

# 11. Project Architecture

Organize the project into clear modules.

A suggested structure is:

project/
app.py
requirements.txt
README.md
config.py

```
data/
    generated/

src/
    data_generator.py
    workload.py
    optimizer.py
    baseline.py
    simulator.py
    metrics.py
    scenarios.py
    visualization.py

tests/
    test_optimizer.py
    test_metrics.py
    test_energy_balance.py
    test_workloads.py

results/
```

You may modify this structure if a different design is more appropriate.

Separate data generation, optimization, simulation, evaluation, and visualization.

Avoid putting the entire application in one Python file.

# 12. Development Instructions

Follow this development sequence:

Phase 1: Inspect the current project directory and understand the existing environment. Do not overwrite existing files without checking them.

Phase 2: Implement the synthetic data generator and baseline scheduler.

Phase 3: Implement the mathematical optimization model and verify that it produces feasible schedules.

Phase 4: Implement metrics and experimental comparisons.

Phase 5: Implement scenario testing and robustness evaluation.

Phase 6: Build the Streamlit dashboard.

Phase 7: Write automated tests and run them.

Phase 8: Run the entire project, identify bugs, fix them, and verify the main workflow.

Phase 9: Write the README with setup instructions, model assumptions, mathematical formulation, algorithm explanation, metric definitions, limitations, and commands to reproduce the experiments.

Do not stop after generating files. Execute the code and debug it.

If dependencies are missing, update requirements.txt and provide installation commands.

If the optimization model is infeasible, diagnose the reason instead of silently relaxing constraints.

If a solver fails, provide a clear error message and implement a documented fallback only if appropriate.

Do not claim that any feature works until you have tested it.

# 13. Final Deliverables

When finished, provide:

1. A complete working project.
2. A functioning Streamlit dashboard.
3. A mathematical optimization model.
4. A baseline scheduler.
5. An experimental evaluation framework.
6. Results from actual executions.
7. Scenario-based robustness analysis.
8. Automated tests and their results.
9. A README with exact commands to install and run the project.
10. A concise explanation of the algorithm that I can use to prepare for competition judging.

The final solution should prioritize mathematical correctness, reproducibility, transparent assumptions, and credible experimental results over unnecessary complexity.

Start by inspecting the existing project directory. Then implement the system step by step, testing each component before proceeding to the next.
