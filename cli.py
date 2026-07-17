"""Simple CLI wrapper for training, evaluation, and prediction."""
import argparse
from stockgpt_clone.train import train_from_config
from stockgpt_clone.evaluate import evaluate_from_config
from stockgpt_clone.predict import predict_next_from_config


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')

    train_p = sub.add_parser('train')
    train_p.add_argument('--config', default='config/default.yaml')

    eval_p = sub.add_parser('evaluate')
    eval_p.add_argument('--config', default='config/default.yaml')

    pred_p = sub.add_parser('predict')
    pred_p.add_argument('--config', default='config/default.yaml')

    args = parser.parse_args()
    if args.cmd == 'train':
        train_from_config(args.config)
    elif args.cmd == 'evaluate':
        evaluate_from_config(args.config)
    elif args.cmd == 'predict':
        val = predict_next_from_config(args.config)
        print(f"Next price prediction: {val}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

