class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return 'datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        if dataset == 'crack':
            # finetuning
            return '/home/jc/xinrun/EvaData/'
            # full training
            # return '/home/jc/datasettest/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
