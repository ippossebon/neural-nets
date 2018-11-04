from instance import Instance

class FileUtils(object):

    def __init__(self, dataset_file, config_file):
        self.dataset_file = dataset_file
        self.config_file = config_file

    def getDataset(self):
        instances = []

        with open(self.dataset_file) as f:
            data = f.readlines()

            for line in data:
                line = line.strip('\n')
                info = line.split(';')
                attributes = info[0]
                attributes = attributes.split(',')
                attributes_list = []

                for att in attributes:
                    attributes_list.append(float(att))

                instance_class = info[1]
                inst = Instance(attributes=attributes_list, classification=instance_class)
                instances.append(inst)

        return instances

    def getConfigParams(self):
        with open(self.config_file) as f:
            data = f.readlines()

            for line in data:
                pass
