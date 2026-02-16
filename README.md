# Tennis Dashboard (SwingVision)

Dashboard built from SwingVision match exports.

## Run locally
Install dependencies (once you have requirements.txt):
pip install -r requirements.txt

Run:
python tennis_dashboard.py

## Data
This repo ignores *.csv by default. Keep match data local.

#Purpose 
An interactive dashboard that converts SwingVision shot-level exports into actionable performance summaries and court visualizations. The dashboard should support fast review of Serve, Return, Rally outcomes, Shot characteristics (speed/spin/direction), and court positioning (hit/bounce maps + movement proxies).

#Target users Primary: 
player + analyst (self-scouting, training focus, match review) Secondary: coach (tactical patterns, serve/return tendencies, positioning)

#Data inputs Input format: 
CSV exported from SwingVision with shot-level row. Identifiers: Set, Game, Point, Shot, Player, Winner Shot metadata: Type (e.g., first_serve, second_serve, first_return, return_plus_one), Stroke (e.g., Forehand, Backhand, Volley), Spin, Direction, Result (In vs not In) Speed: Speed (MPH) Court coordinates (standardized): hit_x_std, hit_y_std, bounce_x_std, bounce_y_std Zones/depth: Hit Zone, Bounce Zone, Hit Depth, Bounce Depth

#Core dashboard sections 
A. Player & Match Overview Filters: Player (including “Opponent”), Set/Game/Point range, Result (In/Out), Stroke, Type KPI cards (per selected player): total points, points won %, winners, unforced errors, net points won %, aces, double faults

B. Serve Module Aces (operational: point length = 1, first_serve in) Double faults (second_serve not In) 1st serve in %, 2nd serve in % 1st serve won %, 2nd serve won %

C. Return Module Return winners Return forced errors (return shot followed by the opponent's miss within the same point) Second serve return missed

D. Rally & Outcomes Module Winners Unforced errors Net points won % All points won % Rally length analytics: Shot-count distribution per point: 1, 2, 3, 4, 5–6, 7+ shots “Most common rally length” overall “Most common rally length when winning vs losing” per player 1st return won %, 2nd return won %

E. Speed Module Serve speeds by Type (first/second serve) Rally-stroke speeds by Stroke (FH/BH/Volley/etc.) Visualization: per-stroke mean with min–max bands

F. Forehand / Backhand Profiles Spin distribution (% + counts) Direction distribution (% + counts) Visual: grouped bar charts (player vs player)

G. “Serve +1” and “Return +1” Patterns (2-shot points) “Serve +1” points: focal player has exactly two shots {first_serve, serve_plus_one} “Return +1” points: focal player has exactly two shots {first_return, return_plus_one} Visualization: win/loss donut (inner ring) with Deuce/Ad breakdown (outer ring)

H. Court Visualizations Hit-position half-court plot using hit_x_std, hit_y_std (meters; baseline y=0, net y=11.885; singles lines ±4.115; doubles ±5.485; service line y≈5.485) Bounce-position half-court plot using bounce_x_std, bounce_y_std

I. Positioning / Movement Proxies Movement summary by player computed from within-point sequential distances between hit positions: average distance per shot, total distance
