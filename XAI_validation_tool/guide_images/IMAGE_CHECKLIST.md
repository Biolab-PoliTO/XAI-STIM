# User Guide - Image and GIF Checklist

This document lists all the images and GIFs needed to complete the interactive User Guide for the XAI-STIM Validation Tool.

## How to Use This Checklist

1. Create screenshots/GIFs according to the descriptions below
2. Save each file with the **exact filename** specified (case-sensitive)
3. Place all files in a single folder
4. When opening the User Guide, select this folder to automatically load all images

---

## Screenshots (PNG/JPG format recommended)

### General Interface

| # | Filename | Description |
|---|----------|-------------|
| 1 | `main_interface.png` | Overview screenshot of the main validation tool interface showing the title, progress bar, and first section (Load MER Data) |
| 2 | `opening_file.png` | Screenshot showing the HTML file in file explorer or browser, demonstrating how to open the tool |

---

### Step 1: Load Data

| # | Filename | Description |
|---|----------|-------------|
| 3 | `load_data.png` | Screenshot of the "Load MER Data" section showing the file selection button and instructions |
| 4 | `data_loaded.png` | Screenshot showing the data preview after successful loading, with statistics cards displaying trajectory count, profiling questions, and survey questions |

---

### Step 2: Profile

| # | Filename | Description |
|---|----------|-------------|
| 5 | `profile.png` | Screenshot of the participant profile form showing Participant ID, Years of Experience, and Role dropdown fields |
| 6 | `ai_questions.png` | Screenshot showing the 6 AI Attitude Questions with Yes/No radio button options |

---

### Step 3: Trajectory Selection

| # | Filename | Description |
|---|----------|-------------|
| 7 | `select_trajectory.png` | Screenshot of trajectory selection screen showing multiple trajectory items in the list, with one highlighted/selected |

---

### Step 4: Manual Annotation

| # | Filename | Description |
|---|----------|-------------|
| 8 | `annotation_interface.png` | Full screenshot of the manual annotation section showing the signal plot with multiple channels/traces, axes labels, and some example annotations |
| 9 | `annotation_controls.png` | Close-up screenshot of the control panel showing Annotation Mode checkbox, Erase Mode checkbox, Amplitude Scale slider, and Clear All button |

---

### Step 5: XAI Review

| # | Filename | Description |
|---|----------|-------------|
| 10 | `xai_interface.png` | Full screenshot of the XAI review section showing signal with Score-CAM heatmap overlay, probability bars on the right, and user annotations |
| 11 | `heatmap_explanation.png` | Screenshot focusing on the heatmap visualization, clearly showing the color gradient (red/yellow for high influence, blue/purple for low influence) |
| 12 | `probability_bars.png` | Close-up screenshot of the channel probability bars on the right side of the plot, showing green (>50%), gold (>45%), and red (<45%) bars with numerical values |
| 13 | `xai_controls.png` | Screenshot of the XAI control panel showing: Annotation Mode, Erase Mode, Show My Annotations, Show XAI Heatmap, Heatmap Opacity slider, and Amplitude Scale slider |

---

### Step 6: Survey

| # | Filename | Description |
|---|----------|-------------|
| 14 | `survey.png` | Screenshot of the survey section showing the trajectory context info and the 4 Likert scale questions with all 6 response options visible |

---

### Step 7-9: Progress, Final Survey, Export

| # | Filename | Description |
|---|----------|-------------|
| 15 | `progress_tracking.png` | Screenshot of trajectory selection showing progress (e.g., "2 of 5 completed"), with some trajectories marked as completed with checkmarks and grayed out |
| 16 | `final_survey.png` | Screenshot of the final survey section showing the 6 overall evaluation questions with Likert scale options |
| 17 | `export.png` | Screenshot of the export section showing the three statistics cards (Trajectories Annotated, Total Annotations, Survey Responses) and the JSON preview textarea with Download Results button |

---

## GIFs/Animated Screenshots

| # | Filename | Description |
|---|----------|-------------|
| 18 | `trajectory_select.gif` | Animated GIF showing the process of clicking on a trajectory item (highlighting it in green) and then clicking the "Annotate Trajectory" button |
| 19 | `create_annotation.gif` | Animated GIF demonstrating click-and-drag to create an annotation on the signal plot: mouse click, drag across time, release, resulting green rectangle appears |
| 20 | `adjust_amplitude.gif` | Animated GIF showing the amplitude scale slider being adjusted and the signal traces visually changing height/amplitude in real-time |
| 21 | `toggle_heatmap.gif` | Animated GIF showing toggling the "Show XAI Heatmap" checkbox on/off and/or adjusting the heatmap opacity slider, demonstrating the heatmap appearing/disappearing or changing transparency |

---

## Image Creation Tips

### For Screenshots:
- Use a browser window at **1280x720 or 1920x1080** resolution for consistency
- Ensure all text is **readable** (avoid tiny fonts)
- **Crop** screenshots to show relevant content (remove unnecessary browser chrome when possible)
- Use a **clean background** (close other applications/tabs visible in screenshots)
- For control panels and close-ups, focus on the specific UI elements

### For GIFs:
- Keep GIFs **short** (2-5 seconds optimal)
- Use **smooth mouse movements** (not too fast)
- Record at **10-15 fps** for smaller file sizes
- **Loop** the animation 2-3 times if the action is very quick
- Recommended tools: ScreenToGif (Windows), LICEcap, or Peek (Linux)
- Keep file sizes under **5MB** per GIF when possible

### File Naming:
- Use **exact filenames** as listed above (case-sensitive)
- All screenshots: `.png` format preferred (better quality for UI)
- All animations: `.gif` format
- Alternative formats accepted: `.jpg`, `.jpeg`, `.webp`

---

## Sample Data for Screenshots

To create realistic screenshots, you may need to:

1. **Load sample trajectories** using the provided `trajectory_T01.json` and `trajectory_T02.json` files
2. **Fill in profile data** with realistic values:
   - Participant ID: `221225_1030` (or current date/time)
   - Years of Experience: `8`
   - Role: `Neurosurgeon`
3. **Create sample annotations** at various time ranges and depths
4. **Answer some survey questions** to show completed state

---

## Checklist Progress

Use this checklist to track your progress:

- [ ] 1. main_interface.png
- [ ] 2. opening_file.png
- [ ] 3. load_data.png
- [ ] 4. data_loaded.png
- [ ] 5. profile.png
- [ ] 6. ai_questions.png
- [ ] 7. select_trajectory.png
- [ ] 8. annotation_interface.png
- [ ] 9. annotation_controls.png
- [ ] 10. xai_interface.png
- [ ] 11. heatmap_explanation.png
- [ ] 12. probability_bars.png
- [ ] 13. xai_controls.png
- [ ] 14. survey.png
- [ ] 15. progress_tracking.png
- [ ] 16. final_survey.png
- [ ] 17. export.png
- [ ] 18. trajectory_select.gif
- [ ] 19. create_annotation.gif
- [ ] 20. adjust_amplitude.gif
- [ ] 21. toggle_heatmap.gif

---

## Total Required

- **Screenshots:** 17
- **GIFs:** 4
- **Total:** 21 visual assets

---

## Once Complete

1. Place all 21 files in a dedicated folder (e.g., `user_guide_images`)
2. Open `USER_GUIDE.html` in your browser
3. Click "Select Image Folder" and choose your folder
4. All images will automatically populate in the guide
5. Review the guide to ensure all images display correctly

---

**Last Updated:** December 22, 2025
