# backup history

## 2020-12-07
Config for first flight

**2020-12-07: 1st flight**
- Keeping the altitude with the barometer works much better now that it's covered with foam.
- The ALTHOLD flightmode works quite well. The altitude changes, but quite slowly. 
- When the drone enters ALTHOLD, it does a little "jump" of about 20 centimeters.
- When the throttle is idle, the drone stands on the ground with the front feet 2-3 millimeters in the air (the back feet are pushed down by the weight of the battery). Thus a ```throttle_idle``` value of 5 might be better.

## 2020-12-09
Config with added opflow/lidar sensor

**2020-12-09: 1st flight**
- Flight modes
   - 1. no hold
   - 2. alt & pos hold
   - 3. alt & pos hold + surface
- When switching into mode 2 the drone does a little "jump". Then it holds position quite well, but slowly decreases altitude until it lands. This might be due to the battery being not fully charged, I'll try this with a charged battery.
- Pos hold worked much better on a contrast-rich carpet.
- Surface didn't really change anything, which was to be expected.

## INAV_cli_diff_all
Version control done by git. See commit messages from now on.
