"""
MIT License

Copyright (c) 2023 bmika1

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import platform
import time
import webbrowser
import tempfile
import os


class BrowserNotFoundException(Exception):
    def __init__(self, browser: str):
        super(BrowserNotFoundException, self).__init__(
            f"Browser not found: {browser}")


class UnknownBrowserException(Exception):
    def __init__(self, e: Exception):
        super(UnknownBrowserException, self).__init__(
            f"Unknown webbrowser exception: {str(e)}"
        )


def _open_in_browser(file_path: str, browser: str | None = None) -> None:
    """
    Open the specified file path in a web browser.

    Args:
        file_path (str): The path of the file to open.
        browser (str | None, optional): The web browser to use (i.e. "chrome", "safari").
            If provided, the HTML content will be opened using the specified browser.
            If not provided or set to None, the default browser will be used.
    """
    if browser:
        try:
            client = webbrowser.get(browser)
        except webbrowser.Error as e:
            if "could not locate runnable browser" in str(e):
                raise BrowserNotFoundException(browser)
            raise UnknownBrowserException(e)
    else:
        client = webbrowser
    if not file_path.startswith("file:///"):
        file_path = f"file:///{file_path}"
    client.open_new(file_path)


def _handle_open_from_temp(html_string: str, browser: str | None = None) -> None:
    """
    Handle opening HTML content from a temporary file in a web browser.

    Args:
        html_string (str): The HTML content as a string.
        browser (str | None, optional): The web browser to use (i.e. "chrome", "safari").
            If provided, the HTML content will be opened using the specified browser.
            If not provided or set to None, the default browser will be used.
    """
    # Set delete parameter depending on the platform
    autodelete = platform.system() != "Windows"

    with tempfile.NamedTemporaryFile(
        mode="w", delete=autodelete, suffix=".html"
    ) as tmp_file:
        tmp_file.write(html_string)
        file_path = tmp_file.name
        if autodelete:
            _open_in_browser(file_path, browser)
            # Adding a short sleep so that the file does not get cleaned
            # up immediately in case the browser takes a while to boot.
            time.sleep(3)
    if not autodelete:
        _open_in_browser(file_path, browser)
        time.sleep(3)
        os.unlink(file_path)  # Cleaning up the file in case of Windows


def _handle_open_from_regular_file(
    html_string: str, save_path: str, browser: str | None = None
) -> None:
    """
    Handle opening HTML content from a regular file in a web browser.

    Args:
        html_string (str): The HTML content as a string.
        save_path (str): The path to save the HTML content as a file.
        browser (str | None, optional): The executable path of the web browser to use.
            If provided, the HTML content will be opened using the specified browser.
            If not provided or set to None, the default browser will be used.
    """
    with open(save_path, "w") as f:
        f.write(html_string)
    _open_in_browser(save_path, browser)


def render_in_browser(
    html_string: str, save_path: str | None = None, browser: str | None = None
) -> None:
    """
    Render the HTML content in a web browser.

    Args:
        html_string (str): The HTML content as a string.
        save_path (str | None, optional): The path to save the HTML content as a file.
            If provided, the HTML content will be saved to the specified file
            and opened from it. If not provided or set to None, a temporary file
            will be created in the operating system's default temporary directory.
            The temporary file will be removed once the rendering is complete.
            IMPORTANT: Please provide an absolute path to your file.
        browser (str | None, optional): The web browser to use (i.e. "chrome", "safari").
            If provided, the HTML content will be opened using the specified browser.
            If not provided or set to None, the default browser will be used.
    """
    if save_path:
        _handle_open_from_regular_file(html_string, save_path, browser)
    else:
        _handle_open_from_temp(html_string, browser)
