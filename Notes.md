#Training Notes


Instead of setting number of epochs, you need to define number of steps. This is because number of epochs is not failure-friendly in distributed training. You need to adjust number of steps based on batchsize and learning rate. For instance, if you want to process for 100 epochs and you have a 1,000 examples, then for a batchsize of 1,000, number of steps would be 100. For a batchsize of 100, number of steps would be 1,000. Basically, number of steps equal number of epochs multiplied by number of examples divided by batchsize.