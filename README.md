**Transfer Learning**

Transfer Learning technique greatly reduce the training complexity and number of trainable parameters will be reduced since we are using the previously trained layers from base model.

Steps for Transfer Learning:

**Step-1 : Freeze the weights of base model:**

Freeze the weights of previously trained layers except for output layer.
to check trainable status of model:

```
for layer in base_model.layers:
        print(layer.trainable)
```
to freeze weights of layers:
```
for layer in base_model.layers[:-1]:
        layer.trainable = False
```
**Step-2: select layers from base model**
```
base_layer = base_model.layers[:-1]
```
**Step-3: Creating new model using previously trained layers**
```
new_model = tf.keras.model.Sequential(base_layer)
```
**Step-4: add new layer**
```
new_model.add(
        tf.keras.layers.Dense(2,activation="softmax",name="output_layer")
)
```
**Step-5: Now new model is ready for training and follow the steps as usual**
```
LOSS = "sparse_categorical_crossentropy"
OPTIMIZER = "SGD"
METRICS = ["accuracy"]

new_model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
new_model.summary()
history = new_model.fit(X_train, y_train_bin, epochs=10, validation_data=(X_valid, y_valid_bin), verbose=2)
new_model.save(model_file_path)
```