
default = {
        'hidden_layer_1': [[192, 1024], [1024]],
        'hidden_layer_2': [[1024, 512], [512]],
        'read_out': [[512, n], [n]],
        'alpha': 1e-3,
        'test_size': .1,
        'batch_size': 192,
        'num_epochs': 1000,
        'drop_out': .3
    }

ensemble_hyperparams = {

    0: {
        'hidden_layer_1': [[192, 1024], [1024]],
        'hidden_layer_2': [[1024, 512], [512]],
        'read_out': [[512, n], [n]],
        'alpha': 1e-4,
        'test_size': .20,
        'batch_size': 250,
        'num_epochs': 2000,
        'drop_out': .3
    },
    1: {
        'hidden_layer_1': [[192, 1024], [1024]],
        'hidden_layer_2': [[1024, 512], [512]],
        'read_out': [[512, n], [n]],
        'alpha': 5e-5,
        'test_size': .20,
        'batch_size': 200,
        'num_epochs': 5000,
        'drop_out': .3
    },
    2: {
        'hidden_layer_1': [[192, 1024], [1024]],
        'hidden_layer_2': [[1024, 512], [512]],
        'read_out': [[512, n], [n]],
        'alpha': 1e-3,
        'test_size': .10,
        'batch_size': 200,
        'num_epochs': 1500,
        'drop_out': .3
    },
    3: {
        'hidden_layer_1': [[192, 1024], [1024]],
        'hidden_layer_2': [[1024, 512], [512]],
        'read_out': [[512, n], [n]],
        'alpha': 5e-5,
        'test_size': .20,
        'batch_size': 200,
        'num_epochs': 5000,
        'drop_out': .3
    },
    4: {
        'hidden_layer_1': [[192, 1024], [1024]],
        'hidden_layer_2': [[1024, 512], [512]],
        'read_out': [[512, n], [n]],
        'alpha': 1e-4,
        'test_size': .15,
        'batch_size': 192,
        'num_epochs': 3000,
        'drop_out': .3
    }
}
