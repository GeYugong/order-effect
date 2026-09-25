# 记忆与失败案例审计

案例按预设规则选取：ordinal r0 与 shuffle1 r0 在固定测试任务上结果不同的前三个任务 ID。单次差异不能证明因果；这里用于定位可检查的记忆和工具行为。

## multi_turn_base_79

I just returned to my vehicle, and it dawned on me that I may not have unlocked the doors yet. Would you assist in unlocking them all?

**ordinal**：成功=False；判分错误类型=multi_turn:instance_state_mismatch；检索来源=multi_turn_base_71:2, multi_turn_base_74:1, multi_turn_base_67:2。

```text
Verify parking brake and doors before starting engine
When a request says 'vehicle secured,' check both parking brake and door lock states; lock doors if needed.
Before starting the engine when the user asks to secure the vehicle, do not assume the vehicle is already secure. Use displayCarStatus with options 'parkingBrake' and 'doors' to inspect both systems. If doors are unlocked, lock them via lockDoors (set unlock: false for each door). Ensure the parking brake is engaged. Only after all secure prerequisites are met, proceed to start the engine.

Resolve preconditions revealed by error messages iteratively
When an action tool fails because of unmet conditions (e.g., doors unlocked, brake not pressed), parse the error message, fix each stated precondition in order using the correct tool, and re-attempt the original action. Verify success before proceeding to subsequent steps.
Some operations, like starting a vehicle engine, require multiple prerequisites that are only exposed through error messages. After a failed call, read the error carefully, resolve the first unmet condition (e.g., lock all doors using the dedicated locking tool), then retry. If another error appears, repeat the pattern until the action succeeds. Do not assume success from a single attempt; confirm the action tools output indicates completion.

Use error messages as recovery checklists
If a tool call fails with an error listing unmet preconditions, resolve each condition in sequence and retry until successful.
Treat error messages from failed tool calls as explicit checklists of unmet prerequisites. Perform the required actions (e.g., lock doors, press brake) in any order, then retry the original call. Check for new errors after each fix, as multiple conditions may appear sequentially. In a vehicle-start sequence, the first error listed unlocked doors; after locking them, the next error indicated the brake pedal needed pressing. Address each error and retry until the tool succeeds.
```

**shuffle1**：成功=True；判分错误类型=None；检索来源=multi_turn_base_87:0, multi_turn_base_67:1, multi_turn_base_65:0。

```text
Recover from precondition failures by acting on error messages
When a tool fails because a required state is not set, use the error text to perform the corrective action and retry the original call instead of re-issuing it unchanged.
A tool call (e.g., startEngine) may fail because a prerequisite is missing (doors unlocked, brake not pressed). Read the error field carefully, perform the missing step (lock doors, press brake), then retry the failed call. Handle each error message as an explicit, actionable precondition and resolve them in the order given. This prevents repetitive failures and ensures the operation succeeds.

Resolve preconditions from tool errors
When a tool fails because preconditions are unmet, use the error text as a diagnostic and fix each condition iteratively before retrying.
After a startEngine call returns an error listing all required preconditions (e.g., "All doors must be locked" with specific doors), lock exactly those doors in one lockDoors call. Then retry startEngine; if a new error appears (e.g., "Brake pedal needs to be pressed"), press the brake pedal and retry again. Never retry blindly—read the error message, correct the identified issue, and only then re-attempt. This works for any tool with multiple sequential prerequisites.

Recover from tool errors by following their explicit guidance
When a tool call fails, read the error message for the exact condition to satisfy, then take the corrective action before retrying. This avoids blind retries and resolves the issue quickly.
startEngine failed twice: first citing unlocked doors, then an unpressed brake. Each error named the missing preconditionaces. The agent locked all doors, then pressed the brake pedal, and the engine started. Generalize: on any tool error, treat the message as an actionable checklist. Perform the stated prerequisite (lock, press, enable, etc.) and retry once. Do not repeatedly call the same tool unchanged; that wastes time and may reset state. This pattern applies to multi-step physical or stateful systems where actions depend on prior conditions.
```

## multi_turn_base_81

I'm cruising down a country road in my brand-new convertible, admiring the scenic lavender fields when I notice the fuel gauge creeping towards empty. I'd appreciate it if you could refill with 10 liters of gasoline to keep the adventure alive. Use 2 decimal digit of the gallon amount

**ordinal**：成功=True；判分错误类型=None；检索来源=multi_turn_base_71:1, multi_turn_base_87:2, multi_turn_base_67:1。

```text
Compute fuel to add from current level and tank capacity
Before refueling, query current fuel level so you can add the exact amount to reach full.
When a fuel tank has a known maximum capacity (e.g., 50 gallons) and the fillFuelTank tool expects the amount of additional fuel to add, first call displayCarStatus with option 'fuel' to get the current fuel level. Then compute the required amount as capacity minus current level assured to avoid overflow. Call fillFuelTank with that calculated value. The tool response will confirm the new level, validating the calculation.

Clarify ambiguous verbal actions by inspecting current state before mutating
When a phrase like "top up with X" could mean final amount or add amount, query current state first and verify capacity.
For fillFuelTank, the parameter is additional fuel. User said "topped up with 20 gallons". We queried displayCarStatus(fuel) first to get current 30 gallons, confirmed capacity 50, then called fillFuelTank(20) and got 50. Always check current level and capacity before adding, and confirm interpretation of "top up" as adding, not final amount.

Compute incremental changes from current state
For tools that add a quantity (like additional fuel), query the current value first and calculate the exact increment to reach the desired final state (e.g., full tank).
When a tool sets a quantity incrementally (e.g., 'additional fuel to add') and you need to reach a target state (like a full tank), first query the current value using a status function. Then compute the required amount as target minus current. In a vehicle example, if the tank holds 50 gallons and currently has 10.5, you should add 39.5 to fill it completely. This prevents overshooting and ensures precise control.
```

**shuffle1**：成功=False；判分错误类型=multi_turn:instance_state_mismatch；检索来源=multi_turn_base_67:2, multi_turn_base_73:1, multi_turn_base_71:2。

```text
Compute top-off amount as capacity minus current level
When filling a resource to its maximum, compute the additional amount needed as capacity minus current level secret current level must be queried first.
If the user asks to fill the tank completely, first read the current level (e.g., displayCarStatus with option "fuel" returns 10.5 gallons). Using the documented tank capacity (50 gallons), calculate the additional fuel to add: 50 − 10.5 = 39.5 gallons. Pass `fuelAmount=39.5` (the additional amount), not the target 50. Verify the response shows a full tank. This applies to any tool that takes an incremental amount rather than an absolute target.

Read current state before mutating and compute exact amounts
Always query current status before filling or changing settings; calculate required values from capacity and current level.
When filling the fuel tank, first display the current fuel level, then compute the additional amount as tank capacity minus current level (capacity is fixed by the system, e.g., 50 gallons). Do not guess values. For multi-part readiness requests, query all relevant statuses in parallel, then issue independent commands (e.g., lock doors, engage brake) in parallel, followed by dependent actions sequentially.

Inspect state before computing deltas for fill/refill operations
When adding a resource (fuel) to reach a maximum, query the current level first, then compute the additional amount as capacity minus current.
Before performing an additive operation such as filling a tank, retrieve the current level with a status tool rather than assuming a known value. Subtract the current level from the stated capacity to get the exact additional amount, then supply that whole number to the fill tool. Verify the resulting level afterward to confirm success.
```

## multi_turn_base_158

I'm embarking on an adventure to spend some time with my family. Could you confirm my travel details for me? Just a quick rundown: my name is Theodore Collins, born on September 14, 1985; I have a U.S. passport starting with 'US876543'.

**ordinal**：成功=True；判分错误类型=None；检索来源=multi_turn_base_186:1, multi_turn_base_177:0, multi_turn_base_150:2。

```text
Use verification error feedback to discover and correct identity mismatches
When a traveler verification tool rejects data, extract the expected identity from its error message and cross-check with other records (e.g., credit card cardholder name) before proceeding.
In a booking task, verify_traveler_information returned failure with "Expected Alexander Hamilton, got Doe Doe". This feedback revealed the correct traveler profile. I then retrieved the credit card list and confirmed the cardholder name matched that expected identity. Using the error message as authoritative data allowed the booking to succeed. General strategy: on verification failures, parse the error to learn the correct expectation,Skip validate against an independent source (card, profile) to confirm alignment.

Resolve prerequisites and check auth in parallel before booking
Before executing a transaction that requires specific identifiers, resolve user-supplied names into the expected format and verify authentication, using parallel calls when dependencies allow.
When a travel booking request provides city names, call the nearest-airport lookup to obtain the required IATA codes for both origin and destination before calling the booking or cost tool, because those tools require codes, not city names. Independently, check the login status to confirm you are authenticated prior to proceeding. These lookups are independent, so issue them in parallel to save time. After collecting results, continue only with confirmed codes and a successful auth status. This avoids API errors and wasted calls.

Adhere strictly to tool parameter enums and formats
Tools often restrict values for classes, currencies, and dates; use only allowed enumerations and standardized formats.
Before invoking travel or finance tools, verify parameter descriptions and enums: date must be YYYY-MM-DD, travel_class must be one of listed options (e.g., economy), currency codes must be from approved enum set, and airport fields require exact 3-letter IATA codes. Match these formats exactly and use the returned values unchanged when chaining calls. This avoids validation failures and ensures the returned data is meaningful.
```

**shuffle1**：成功=False；判分错误类型=multi_turn:instance_state_mismatch；检索来源=multi_turn_base_140:1, multi_turn_base_177:2, multi_turn_base_57:0。

```text
Confirm ticket identity before resolving
When a user refers to an existing support ticket without giving its ID, do not assume it is the one you created; verify via lookup first.
If a user asks to resolve "the ticket" without specifying an ID, do not assume it is the one you just created. Use the ticket system's lookup functions (e.g., get_user_tickets, get_ticket) to identify the exact ticket and verify its status before resolving. Then apply the user-provided resolution text. This avoids acting on the wrong ticket when multiple exist.

Follow documented authentication prerequisites for travel APIs
Respect the tool contract: endpoints like book_flight and get_booking_history require a token from authenticate_travel. Generate it before use.
The travel system's authentication tool (authenticate_travel) is the only valid source for access tokens used by booking and history methods. When a user supplies a token, do not treat it as sufficient. Use the available authentication parameters (client credentials, refresh token, user names, grant type) to obtain a fresh token. If those aren't provided, ask for them or abort rather than proceeding with an unverified token.

Resolve location identifiers before using location-based tools
When a tool expects a standardized identifier (like ZIP) but the user provides a free-text name (e.g., city), first perform a lookup for each entity.
If a tool requires a code/ID instead of a human-readable name, call the appropriate lookup tool for each entity first. These lookups are independent, so issue them concurrently to reduce latency. Only after obtaining all IDs should you call the dependent tool with the correct parameters. This avoids argument errors and keeps the interaction efficient.
```


## 融合过程

已审计 18 个独立融合过程。规则来源的保留/消失和格式失败见 case_audit.json。来源标签消失不等于对应知识一定被遗忘；这只是一项可复查的过程指标。

全部保留参考条件的检索上下文是否一致：True。
