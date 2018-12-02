from instance import Instance

class FileUtils(object):

    def __init__(self, dataset_file=None, config_file=None):
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
                output = instance_class.split(',')
                output_list = []
                for item in output:
                    output_list.append(float(item))

                inst = Instance(attributes=attributes_list, classification=output_list)

                instances.append(inst)

        return instances

    def getConfigParams(self):
        content = []
        param_list = []

        with open(self.config_file) as f:
            content = f.readlines()

        content = [x.strip() for x in content]
        content_len = len(content)

        param_list.append(float(content[0]))

        for data in range(1, content_len):
            param_list.append(int(content[data]))

        return param_list
