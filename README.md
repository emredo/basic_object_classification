# Classification with Support Vector Machine
<section>
This repository is about traffic sign classification. YOLO can classify traffic signs well, but in some cases YOLO can easily confuse same classses. This repository tries to overcome this problem.
</section>

### Requirements
<section>
scikit-learn  (latest version is preferred)

opencv (to load image, PIL can be used also)

Sample data.npy file can be downloaded with this link: ``` https://drive.google.com/file/d/1erwsyowce69Z1BqcBM4QZiPFuYgScwsc/view?usp=sharing ```
</section>

### Training
<section>
Training process is very easy to understand. First the train data must be prepared. Data scheme is given following.

X axis is 64 * 64 * 3 length row. Y axis is one hot encoded 0 and 1.

For instance, if the sign is no-right Y axis is (0,1). If the sign is no-left the Y axis is (1,0).

After that, if x and y is merged vertically, 12290 length rows can be created.
</section>

### Inference
<section>
Inference code is easy to understand, all details given in .py file.
</section>

### *If you have any questions about repository, feel free to create issue.*