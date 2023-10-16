def main(args):
    pass

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to the model.")
    args = parser.parse_args()

    main(args)
