from setup_utils import load_train_yaml

def main(args):
    yaml_data = load_train_yaml(args.dataset, "Sync")

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        choices=["cora", "citeseer", "amazon_photo", "amazon_computer"])
    args = parser.parse_args()

    main(args)
