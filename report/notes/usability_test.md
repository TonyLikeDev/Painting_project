# Usability test of the web app (Week 8): protocol

The plan for Week 8 asks for a usability test with 5 to 8 users who are not developers, measuring task completion, the time to
the first painting and the problems found, and then fixing the most important problems. This note is the protocol; the
participant sheet in Vietnamese is `usability_test_phieu_vi.md`; the results go to a new folder
`experiments/<date>_usability/` (copy the three empty CSV files of `experiments/usability_test_template/`), summarized by
`scripts/usability_summary.py`.

## Participants and set-up

- **5 to 8 people** who have never seen the app and do not program: classmates from other faculties, family, friends. Note their
  age range and how often they use a computer (daily / weekly / rarely); no names in the data, only P1, P2, ...
- **One laptop** with the app running (double-click `Start_App.bat`; on the desktop a painting takes about 15 s, on a laptop
  without an NVIDIA GPU about two minutes, so prefer the desktop or tell the participant that waiting is expected), the browser
  open on the page, the window maximized. A folder `Ảnh thử` on the desktop with three photos of different kinds (a portrait,
  a landscape, an object), and the participant may use a photo from their own phone instead.
- **One observer** (the student) with a stopwatch (a phone) and the observation sheet below. About 15 minutes per participant.

## Procedure

1. Explain in one sentence: "This app turns a photo into a painting. I am testing the app, not you; please say aloud what you
   think while you use it." Ask for consent to note what they do and their answers (no photos of them, no recordings unless they
   agree).
2. Give the tasks one at a time, read aloud from the participant sheet. Do not help. If the participant is stuck for more than
   two minutes, or asks, give the smallest hint that unblocks them and note the task as **help**.
3. For each task note: the outcome (`done`, `help`, `failed`), the time in seconds, and every problem you see or hear (hesitations,
   wrong clicks, misunderstandings, comments).
4. After the tasks, the participant fills in the 10 questions of the System Usability Scale and answers the three open questions.

## Tasks

| Task | Instruction to the participant | Success when | Time from / to |
| :--- | :--- | :--- | :--- |
| T1 | Make a painting of the apple photo with oil paint. | the finished painting is on the screen | first look at the page / the painting is finished |
| T2 | Make a watercolour painting of a photo from the folder `Ảnh thử` (or of a photo from your phone). | the watercolour painting of that photo is on the screen | the instruction / finished |
| T3 | Compare your painting with the photo, then look at the painting when it was about half done. | uses the before / after slider and the replay slider | the instruction / both seen |
| T4 | Make a quicker painting of the same photo with fewer strokes, with the marker pen. | a marker painting with fewer strokes, fast setting or a lower count | the instruction / finished |
| T5 | Save the painting on the computer. | the PNG is in the Downloads folder | the instruction / file saved |

T1 measures the time to the first painting, the plan's measure; it includes finding the sample photo, the brush and the Paint
button, and the painting itself (about 15 s on the desktop).

## Measures and analysis

- **Task completion**: per task, how many participants did it alone, with help, or failed.
- **Time to the first painting**: the T1 times; median and range.
- **System Usability Scale** (SUS, Brooke 1996): ten statements answered from 1 (strongly disagree) to 5 (strongly agree); the
  score is $2.5 \sum$ (odd items: answer $- 1$; even items: $5 -$ answer), from 0 to 100; about 68 is the average commonly
  reported for many systems (Sauro 2011), above 80 is good.
- **Problems**: every problem written once per participant with a severity: 1 cosmetic (noticed, no effect), 2 slows the task, 3
  blocks the task or leads to a wrong result. A problem's priority is the number of participants who met it times its worst
  severity; the problems with the highest priority are fixed first, and the fixed version is checked with one more person.

## Data files

Fill in, one row per observation (UTF-8; Excel: "CSV UTF-8"):

- `tasks.csv`: `participant,task,outcome,seconds,notes` (outcome `done`, `help` or `failed`);
- `sus.csv`: `participant,q1,...,q10` (the answers 1 to 5 as given, the script reverses the even items);
- `issues.csv`: `participant,issue,severity` (use the same short wording for the same problem).

Then run `venv/Scripts/python.exe scripts/usability_summary.py experiments/<date>_usability`, which writes `summary.md` with the
tables for the report.

## Observation sheet (one per participant)

| | T1 | T2 | T3 | T4 | T5 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Outcome (done / help / failed) | | | | | |
| Time (s) | | | | | |
| Problems seen or heard | | | | | |

Participant P__ , age range ____ , computer use: daily / weekly / rarely, device of the test: ________ , date ________
