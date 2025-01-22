import twine.__main__
import twine.repository
# When your company enforces traffic inspection and replaces certificates for security,
# sometimes you end up having to use these less-than-ideal workarounds.

# Usage: twine-trusted <args>
# Source: https://github.com/pypa/twine/pull/463

import requests
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)


def disable_server_certificate_validation():
    """Allow twine to just trust the hosts"""
    twine.repository.Repository.set_certificate_authority = lambda *args, **kwargs: None


def main():
    disable_server_certificate_validation()

    # Forward all command-line arguments to the original twine main function
    twine.__main__.main()


if __name__ == '__main__':
    main()
