//importing the required modules
const tf = require('@tensorflow/tfjs-node')
const fs = require('fs');
const csv = require('csvtojson')
// const fetch = require('node-fetch')

//initializing variables globally

let labels_train = []
let labels_val = []
let r_train = []
let r_val = []
let train_x, train_y, val_x, val_y

async function getModel() {
    const input = tf.layers.input({ shape: [48, 48, 1] });
    const conv1 = tf.layers.conv2d({ filters: 64, kernelSize: 5, kernelInitializer: 'glorotNormal', padding: "same", name: "conv_1", activation: "relu" }).apply(input)
    const conv2 = tf.layers.conv2d({ filters: 64, kernelSize: [3, 3], kernelInitializer: 'glorotNormal', name: "conv_2", activation: "relu", padding: "same" }).apply(conv1)
    const max1 = tf.layers.maxPool2d({ poolSize: [2, 2], name: "maxpool1" }).apply((tf.layers.concatenate({ axis: -1 }).apply([conv1, conv2])))

    const conv3 = tf.layers.conv2d({ filters: 64, kernelSize: [1, 1], kernelInitializer: 'glorotNormal', name: "conv_3", activation: "relu" }).apply(max1);
    const conv4 = tf.layers.conv2d({ filters: 128, kernelSize: [5, 5], kernelInitializer: 'glorotNormal', padding: "same", name: "conv_4", activation: "relu" }).apply(conv3);
    const conv5 = tf.layers.conv2d({ filters: 128, kernelSize: [3, 3], kernelInitializer: 'glorotNormal', name: "conv_5", activation: "relu", padding: "same" }).apply(conv4);
    const max2 = tf.layers.maxPool2d({ poolSize: [2, 2], name: "maxpool2" }).apply((tf.layers.concatenate({ axis: -1 }).apply([conv3, conv4, conv5])))

    const conv6 = tf.layers.conv2d({ filters: 128, kernelSize: [1, 1], kernelInitializer: 'glorotNormal', name: "conv_6", activation: "relu" }).apply(max2);
    const conv7 = tf.layers.conv2d({ filters: 128, kernelSize: [3, 3], kernelInitializer: 'glorotNormal', name: "conv_7", activation: "relu" }).apply(conv6)
    const avg_pool = tf.layers.averagePooling2d({ poolSize: [2, 2], name: "avg_pool1" }).apply(conv7)
    const flat = tf.layers.flatten().apply(avg_pool)
    const dense1 = tf.layers.dense({ units: 512, kernelInitializer: 'glorotNormal', activation: "relu", name: "dense_1" }).apply(flat)
    const dense2 = tf.layers.dense({ units: 64, kernelInitializer: 'glorotNormal', activation: "relu", name: "dense_2" }).apply(dense1)
    const dense3 = tf.layers.dense({ units: 32, kernelInitializer: 'glorotNormal', activation: "relu", name: "dense_3" }).apply(dense2)
    const output = tf.layers.dense({ units: 1, kernelInitializer: 'glorotNormal', activation: "relu", name: "output" }).apply(dense3)
    const model = tf.model({ inputs: input, outputs: output })
    return model;
}


//function to extract the data from the csv file
async function get_Data() {
    const train_data = await csv().fromFile('train_age.csv');

    for (let i = 0; i < train_data.length; i++) {
        d = train_data[i];
        labels_train.push(parseInt(d['age']));

        temp_pixs = d['pixels'].split(' ');
        let temp_int_pixs = [];
        temp_pixs.forEach(pix => {
            pix = parseInt(pix) / 255;
            r_train.push(pix);
        })
    }

    //creating 4d tensor from the r_train and labels train
    train_x = tf.tensor4d(r_train, [14081, 48, 48, 1])
    train_y = tf.tensor2d(labels_train, [14081, 1])

    const val_data = await csv().fromFile('val_age.csv');

    for (let i = 0; i < val_data.length; i++) {
        d = val_data[i];
        labels_val.push(parseInt(d['age']));

        temp_pixs = d['pixels'].split(' ');
        let temp_int_pixs = [];
        temp_pixs.forEach(pix => {
            pix = parseInt(pix) / 255;
            r_val.push(pix);
        })
    }

    val_x = tf.tensor4d(r_val, [4694, 48, 48, 1])
    val_y = tf.tensor2d(labels_val, [4694, 1])
}

(async () => {
    console.log("Initialize");
    let begin = Date.now();
    //logging the timestamp when the execution is initalised
    console.log(begin)
    //extracting the data and logging the time taken
    await get_Data();
    console.log("data extracted in seconds ", (Date.now() - begin) / 60)
    //loading the model architecture and loggging the time taken
    let model = await getModel();
    console.log("model Extracted ", Date.now() - begin)

    //mentioning the optimizer and loss
    model.compile({ optimizer: 'adam', loss: 'meanSquaredError' })

    //training the model and printing the loss and epoch number along with time taken for each epoch and saving the model after each epoch 
    const h = await model.fit(train_x, train_y, {
        batchSize: 32, epochs: 10, verbose: 2, validationData: [val_x, val_y], callbacks: {
            onEpochEnd: async (epoch, log) => {
                console.log(`Epoch ${epoch}: loss = ${log.loss} Time: ${Date.now() - begin}`)
                await model.save("file://./models/my-model-" + epoch)
                console.log("model Trained", Date.now() - begin)
            }
        }
    })

})();
