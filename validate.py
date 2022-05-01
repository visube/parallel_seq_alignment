import sys

def parse_args():
    args = sys.argv
    parsed = {}
    parsed['actual'] = args[args.index('-o') + 1]
    parsed['expected'] = args[args.index('-r') + 1]
    return parsed

def main(args):
    val = validate(args)
    print "Correctness: " + str(val)

def validate(args):
    # Input file
    actual = open(args['actual'], 'r')
    expected = open(args['expected'], 'r')
    actual_lines = actual.readlines()
    expected_lines = expected.readlines()
    for i in range(0, len(actual_lines)): 
        if (actual_lines[i] != expected_lines[i]):
            print "Unmatched row " + str(i)
            return False
    return True

if __name__ == '__main__':
    main(parse_args())