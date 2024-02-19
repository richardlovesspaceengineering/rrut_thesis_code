import datetime


def print_with_timestamp(*args, **kwargs):
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Combine the arguments into a single message string
    combined_message = " ".join(str(arg) for arg in args)

    # Check if the message starts with a newline
    if combined_message.startswith("\n"):
        # Remove the first newline character and print the rest with a newline, timestamp, and text
        formatted_message = f"\n[{timestamp}] {combined_message[1:]}"
    else:
        # Format message normally with the timestamp at the start
        formatted_message = f"[{timestamp}] {combined_message}"

    # Use the built-in print function to output the formatted message
    # kwargs are passed through to allow for customization of print() behavior (e.g., file=, sep=, end=, flush=)
    print(formatted_message, **kwargs)
