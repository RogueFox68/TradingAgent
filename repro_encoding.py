
import sys

def safe_print(*args, **kwargs):
    try:
        # Try normal print first
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # If it fails, encode with replacement
        encoding = sys.stdout.encoding or 'ascii'
        new_args = []
        for arg in args:
            if isinstance(arg, str):
                # Encode to bytes with replacement, then decode back to string
                # This turns unprintable chars into '?'
                cleaned = arg.encode(encoding, errors='replace').decode(encoding)
                new_args.append(cleaned)
            else:
                new_args.append(arg)
        print(*new_args, **kwargs)

# Override built-in print (only for this script scope)
# In the real file we will define safe_print and assign print = safe_print

if __name__ == "__main__":
    # This should fail if we didn't use safe_print and encoding is cp1252
    try:
        print("Test Emoji: 🔬") 
    except UnicodeEncodeError:
        print("Standard print failed as expected.")

    # Now test safe mechanism
    # We can't easily overwrite 'print' in this snippet without recursion if we use 'print' inside.
    # So let's just call safe_print directly to test logic.
    safe_print("Safe Emoji: 🔬")
