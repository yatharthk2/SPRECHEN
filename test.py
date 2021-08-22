import torchtext
torchtext.datasets.Multi30k(root='.data', split=('train_data', 'valid_data', 'test_data'), language_pair=('de', 'en') , fields=(german, english))