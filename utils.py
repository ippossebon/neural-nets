from instance import Instance

class FileUtils(object):

    def __init__(self, filename):
        self.filename = filename

    def getDataset(self):
        instances = []

        with open(self.filename) as f:
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
