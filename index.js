import * as tf from '@tensorflow/tfjs';

// Define a model for linear regression.
const model = tf.sequential();
model.add(tf.layers.dense({units: 100, activation: 'relu', inputShape: [1]}));
model.add(tf.layers.dense({units: 1, activation: 'linear'}));
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// Train the model using the data.
model.fit(xs, ys, {epochs: 10}).then(() => {
  // Use the model to do inference on a data point the model hasn't seen before:
  console.log("here...")
  model.predict(tf.tensor2d([5], [1, 1])).print();
});