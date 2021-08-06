export const IMAGE_H = 28;
export const IMAGE_W = 28;
const IMAGE_SIZE = IMAGE_H * IMAGE_W;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;

const NUM_TRAIN_ELEMENTS = 55000;
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

const MNIST_IMAGES_SPRITE_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

/**
 * A class that fetches the sprited MNIST dataset and provide data as
 * tf.Tensors.
 */
export class MnistData {
  constructor(batch_size) {
      this.batch_size = batch_size;
      this.trainNum = parseInt(NUM_TRAIN_ELEMENTS / batch_size);
      this.testNum = parseInt(NUM_TEST_ELEMENTS / batch_size);
      console.log(this.batch_size, this.trainNum, this.testNum);
  }

  async load() {
    // Make a request for the MNIST sprited image.
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const imgRequest = new Promise((resolve, reject) => {
      img.crossOrigin = '';
      img.onload = () => {
        img.width = img.naturalWidth;
        img.height = img.naturalHeight;

        const datasetBytesBuffer =
            new ArrayBuffer(65000 * IMAGE_SIZE * 4);

        const chunkSize = 5000;
        canvas.width = img.width;
        canvas.height = chunkSize;

        for (let i = 0; i < 65000 / chunkSize; i++) {
          const datasetBytesView = new Float32Array(
              datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4,
              IMAGE_SIZE * chunkSize);
          ctx.drawImage(
              img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width,
              chunkSize);

          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

          for (let j = 0; j < imageData.data.length / 4; j++) {
            // All channels hold an equal value since the image is grayscale, so
            // just read the red channel.
            datasetBytesView[j] = imageData.data[j * 4] / 255;
          }
        }
        this.datasetImages = new Float32Array(datasetBytesBuffer);

        resolve();
      };
      img.src = MNIST_IMAGES_SPRITE_PATH;
    });

    const labelsRequest = fetch(MNIST_LABELS_PATH);
    const [imgResponse, labelsResponse] =
        await Promise.all([imgRequest, labelsRequest]);

    this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

    // Slice the the images and labels into train and test sets.
    this.trainImages =
        this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS, IMAGE_SIZE * NUM_DATASET_ELEMENTS);
    this.trainLabels =
        this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
    this.testLabels =
        this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS, NUM_CLASSES * NUM_DATASET_ELEMENTS);
  }

  getTrainData(){
    let trainDataGenerator = function* (trainImages, trainLabels, trainNum, batch_size){
        for(let i = 0; i < trainNum; i++){
            yield {
                data: trainImages.slice(i * IMAGE_SIZE * batch_size, (i+1) * IMAGE_SIZE * batch_size),
                label: trainLabels.slice(i * NUM_CLASSES * batch_size, (i+1) * NUM_CLASSES * batch_size)
            }
        }
        if(trainNum * batch_size < NUM_TRAIN_ELEMENTS){
            yield {
                data: trainImages.slice(trainNum * batch_size * IMAGE_SIZE),
                label: trainLabels.slice(trainNum * batch_size * NUM_CLASSES)
            }
        }
    }
    return trainDataGenerator(this.trainImages, this.trainLabels, this.trainNum, this.batch_size);
  }

  getTestData(){
    let testDataGenerator = function* (testImages, testLabels, testNum, batch_size){
        for(let i = 0; i < testNum; i++){
            yield {
                data: testImages.slice(i * IMAGE_SIZE * batch_size, (i+1) * IMAGE_SIZE * batch_size),
                label: testLabels.slice(i * NUM_CLASSES * batch_size, (i+1) * NUM_CLASSES * batch_size)
            }
        }
        
        if(testNum * batch_size < NUM_TEST_ELEMENTS){
            yield {
                data: testImages.slice(testNum * batch_size * IMAGE_SIZE),
                label: testLabels.slice(testNum * batch_size * NUM_CLASSES)
            }
        }
        
    }
    return testDataGenerator(this.testImages, this.testLabels, this.testNum, this.batch_size);
  }

}