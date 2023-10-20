from setup_utils import load_train_yaml, set_seed

def main(args):
    model_name = "Async"
    yaml_data = load_train_yaml(args.dataset, model_name)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        choices=["cora", "citeseer", "amazon_photo", "amazon_computer"])
    args = parser.parse_args()

    main(args)
