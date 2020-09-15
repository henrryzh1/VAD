import options
class Config(object):
    def __init__(self, args):
        self.lr = eval(args.lr)
        self.num_iters = len(self.lr)
        self.num_classes = 14
        self.modal = args.modal
        if self.modal == 'all':
            self.len_feature = 2048
        else:
            self.len_feature = 1024
        self.batch_size = args.batch_size
        self.data_path = args.data_path
        self.model_path = args.model_path
        self.output_path = args.output_path
        self.log_path = args.log_path
        self.num_workers = args.num_workers
        self.alpha = args.alpha
        self.class_thresh = args.class_th
        self.model_file = args.model_file
        self.seed = args.seed
        self.feature_fps = 30
        self.num_segments = 400
class_dict = {0: 'Abuse',
                1: 'Arrest',
                2: 'Arson',
                3: 'Assault',
                4: 'Burglary',
                5: 'Explosion',
                6: 'Fighting',
                7: 'RoadAccidents',
                8: 'Robbery',
                9: 'Shooting',
                10: 'Shoplifting',
                11: 'Stealing',
                12: 'Vandalism',
                13: 'Normal'
            }
if __name__ == "__main__":
    args=options.parse_args()
    conf=Config(args)
    print(conf.lr)  

