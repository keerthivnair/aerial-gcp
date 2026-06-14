# Decision Log: Aerial GCP Pose Estimation

## 1. Why a Two-Stage Pipeline?
I decided to split the problem into two parts instead of just using one big model like YOLO. 

The drone images are massive (4000x3000). If I just fed them into YOLO, the images would get squashed down, and I would lose the pixel-perfect accuracy needed for surveying. 
Instead, I used **YOLOv8** just as a "crop finder" to draw a box around the general area of the marker. Once I had that box, I took a high-resolution 512x512 crop of that specific area and passed it into a **U-Net**. The U-Net's job is to look closely at that crop and predict a heatmap where the brightest pixel is the exact dead-center of the GCP. 

I also added a simple classification head to the end of the U-Net so it could predict the shape (Cross, Square, L-Shape) at the same time, saving time instead of running a third model.

## 2. Training Strategy & Data Handling
- **Custom Loss Function:** Most of the image is just empty grass or dirt, and the actual GCP center is only a few pixels. To stop the model from just predicting "background" everywhere, I used a weighted MSE loss that heavily penalizes the model if it misses the GCP center pixels.
- **Handling Class Imbalance:** I noticed there were way more L-Shapes (491) than Crosses (177). If I didn't fix this, the model would just get lazy and guess "L-Shape" all the time. I fixed this by adding class weights to the CrossEntropyLoss, forcing the model to care more when it got a 'Cross' wrong.
- **Splitting the Data:** Drone images taken back-to-back look almost identical. If I put similar images in both the train and validation sets, the model would be cheating. I used `GroupKFold` based on the flight IDs to make sure the training and validation sets were completely separated geographically.

## 3. Main Challenges
- **The "Small Object" Problem:** 
  - **Challenge:** Squashing a massive image down to a standard YOLO size turns the marker into a tiny, unrecognizable blur.
  - **Fix:** I trained YOLOv8 for a long time (300 epochs) to force it to learn those tiny, blurry features.
- **Getting Sub-Pixel Accuracy:**
  - **Challenge:** Standard bounding boxes aren't accurate enough for surveying where you need <10px error.
  - **Fix:** Instead of having the model predict X/Y coordinates directly, I had it generate a 2D Heatmap. It's much easier for a neural network to draw a blurry dot over the center than to guess the exact math coordinates.

## 4. Evaluation
- **Localization:** Achieved > 88% accuracy when measuring if the predicted dot fell within 10px, 25px, or 50px of the real center.
- **Classification:** Achieved an F1-Score of 88.6%, which proves that my class weighting actually worked to balance out the rare shapes.
