pygmi.rsense.render_html
========================

.. py:module:: pygmi.rsense.render_html

.. autoapi-nested-parse::

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



Exceptions
----------

.. autoapisummary::

   pygmi.rsense.render_html.BrowserNotFoundException
   pygmi.rsense.render_html.UnknownBrowserException


Functions
---------

.. autoapisummary::

   pygmi.rsense.render_html.render_in_browser


Module Contents
---------------

.. py:exception:: BrowserNotFoundException(browser: str)

   Bases: :py:obj:`Exception`


   Common base class for all non-exit exceptions.


.. py:exception:: UnknownBrowserException(e: Exception)

   Bases: :py:obj:`Exception`


   Common base class for all non-exit exceptions.


.. py:function:: render_in_browser(html_string: str, save_path: str | None = None, browser: str | None = None) -> None

   Render the HTML content in a web browser.

   :param html_string: The HTML content as a string.
   :type html_string: str
   :param save_path: The path to save the HTML content as a file.
                     If provided, the HTML content will be saved to the specified file
                     and opened from it. If not provided or set to None, a temporary file
                     will be created in the operating system's default temporary directory.
                     The temporary file will be removed once the rendering is complete.
                     IMPORTANT: Please provide an absolute path to your file.
   :type save_path: str | None, optional
   :param browser: The web browser to use (i.e. "chrome", "safari").
                   If provided, the HTML content will be opened using the specified browser.
                   If not provided or set to None, the default browser will be used.
   :type browser: str | None, optional


