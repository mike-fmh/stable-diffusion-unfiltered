from PIL import Image, ImageDraw, ImageFont
from textwrap import wrap
import unicodedata
import re


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def get_y_and_heights(text_wrapped, dimensions, margin, font):
    """Get the first vertical coordinate at which to draw text and the height of each line of text"""
    # https://stackoverflow.com/a/46220683/9263761
    ascent, descent = font.getmetrics()

    # Calculate the height needed to draw each line of text (including its bottom margin)
    line_heights = [
        font.getmask(text_line).getbbox()[3] + descent + margin
        for text_line in text_wrapped
    ]
    # The last line doesn't have a bottom margin
    line_heights[-1] -= margin

    # Total height needed
    height_text = sum(line_heights)

    # Calculate the Y coordinate at which to draw the first line of text
    y = (dimensions[1] - height_text) // 2

    # Return the first Y coordinate and a list with the height of each line
    return (y, line_heights)


FONT_FAMILY = "arial.ttf"
WIDTH = 1920
HEIGHT = 1080
FONT_SIZE = 75
V_MARGIN = 40
CHAR_LIMIT = 1000000
BG_COLOR = "black"
TEXT_COLOR = "white"


infile = None

if infile is not None:
    with open(infile, "r") as f:
        filelines = f.readlines()
        for l in filelines:
            l = l.strip()

            text = l

            # Create the font
            font = ImageFont.truetype(FONT_FAMILY, FONT_SIZE)
            # New image based on the settings defined above
            img = Image.new("RGB", (WIDTH, HEIGHT), color=BG_COLOR)
            # Interface to draw on the image
            draw_interface = ImageDraw.Draw(img)

            # Wrap the `text` string into a list of `CHAR_LIMIT`-character strings
            text_lines = wrap(text, CHAR_LIMIT)
            # Get the first vertical coordinate at which to draw text and the height of each line of text
            y, line_heights = get_y_and_heights(
                text_lines,
                (WIDTH, HEIGHT),
                V_MARGIN,
                font
            )

            # Draw each line of text
            for i, line in enumerate(text_lines):
                # Calculate the horizontally-centered position at which to draw this line
                line_width = font.getmask(line).getbbox()[2]
                x = ((WIDTH - line_width) // 2)

                # Draw this line
                draw_interface.text((x, 950), line, font=font, fill=TEXT_COLOR)

                # Move on to the height at which the next line should be drawn at
                y += line_heights[i]

            # Save the resulting image
            fname = slugify(line)
            img.save(f"{fname}.png")
else:
    text = "our reflections clearly in hollywood"

    # Create the font
    font = ImageFont.truetype(FONT_FAMILY, FONT_SIZE)
    # New image based on the settings defined above
    img = Image.new("RGB", (WIDTH, HEIGHT), color=BG_COLOR)
    # Interface to draw on the image
    draw_interface = ImageDraw.Draw(img)

    # Wrap the `text` string into a list of `CHAR_LIMIT`-character strings
    text_lines = wrap(text, CHAR_LIMIT)
    # Get the first vertical coordinate at which to draw text and the height of each line of text
    y, line_heights = get_y_and_heights(
        text_lines,
        (WIDTH, HEIGHT),
        V_MARGIN,
        font
    )

    # Draw each line of text
    for i, line in enumerate(text_lines):
        # Calculate the horizontally-centered position at which to draw this line
        line_width = font.getmask(line).getbbox()[2]
        x = ((WIDTH - line_width) // 2)

        # Draw this line
        draw_interface.text((x, 950), line, font=font, fill=TEXT_COLOR)

        # Move on to the height at which the next line should be drawn at
        y += line_heights[i]

    # Save the resulting image
    fname = slugify(line)
    img.save(f"{fname}.png")