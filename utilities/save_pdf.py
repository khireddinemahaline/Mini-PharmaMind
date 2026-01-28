import os
import re
from datetime import datetime
from fpdf import FPDF


class EnhancedPDF(FPDF):
    """Enhanced PDF class with headers, footers, and better formatting."""

    def __init__(self, title: str = "Research Report", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report_title = title
        self.page_count = 0

    def header(self):
        """Add header to each page."""
        # Logo or icon placeholder (you can add an actual image here)
        self.set_font("Arial", "B", 16)
        self.set_text_color(67, 97, 238)  # Blue color
        self.cell(0, 10, "Agentic Pharma Research", ln=True, align="C")

        # Add a line
        self.set_draw_color(200, 200, 200)
        self.line(10, 20, 200, 20)
        self.ln(5)

    def footer(self):
        """Add footer to each page."""
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(128, 128, 128)

        # Page number
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

        # Timestamp
        self.set_y(-15)
        self.set_x(10)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.cell(0, 10, f"Generated: {timestamp}", align="L")

    def chapter_title(self, title: str):
        """Add a styled chapter title."""
        self.set_font("Arial", "B", 16)
        self.set_fill_color(240, 240, 255)
        self.set_text_color(40, 40, 100)
        self.cell(0, 12, title, ln=True, fill=True)
        self.ln(4)

    def section_title(self, title: str):
        """Add a styled section title."""
        self.set_font("Arial", "B", 14)
        self.set_text_color(60, 60, 150)
        self.cell(0, 10, title, ln=True)
        self.ln(2)

    def subsection_title(self, title: str):
        """Add a styled subsection title."""
        self.set_font("Arial", "B", 12)
        self.set_text_color(80, 80, 160)
        self.cell(0, 8, title, ln=True)
        self.ln(1)

    def body_text(self, text: str):
        """Add body text with proper formatting."""
        self.set_font("Arial", "", 11)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def bullet_list(self, items: list):
        """Add a bullet list."""
        self.set_font("Arial", "", 11)
        self.set_text_color(40, 40, 40)
        for item in items:
            # Add bullet point (using ASCII-safe character)
            x_before = self.get_x()
            y_before = self.get_y()
            self.cell(5, 6, "-", ln=0)
            self.set_x(x_before + 8)
            self.multi_cell(0, 6, item.strip())
        self.ln(2)

    def numbered_list(self, items: list):
        """Add a numbered list."""
        self.set_font("Arial", "", 11)
        self.set_text_color(40, 40, 40)
        for i, item in enumerate(items, 1):
            x_before = self.get_x()
            y_before = self.get_y()
            self.cell(8, 6, f"{i}.", ln=0)
            self.set_x(x_before + 10)
            self.multi_cell(0, 6, item.strip())
        self.ln(2)

    def code_block(self, code: str):
        """Add a code block with background."""
        self.set_font("Courier", "", 9)
        self.set_fill_color(245, 245, 245)
        self.set_text_color(50, 50, 50)
        lines = code.split("\n")
        for line in lines:
            self.cell(0, 5, line, ln=True, fill=True)
        self.ln(3)

    def info_box(self, text: str, box_type: str = "info"):
        """Add an info/warning/success box."""
        # Set colors based on type (using ASCII-safe icons)
        colors = {
            "info": {"bg": (220, 240, 255), "text": (0, 80, 160), "icon": "[i]"},
            "warning": {"bg": (255, 245, 220), "text": (160, 100, 0), "icon": "[!]"},
            "success": {"bg": (220, 255, 220), "text": (0, 120, 0), "icon": "[OK]"},
            "error": {"bg": (255, 220, 220), "text": (160, 0, 0), "icon": "[X]"},
        }

        color = colors.get(box_type, colors["info"])
        self.set_fill_color(*color["bg"])
        self.set_text_color(*color["text"])
        self.set_font("Arial", "B", 11)

        # Draw the box
        self.multi_cell(0, 6, f"{color['icon']} {text}", fill=True, border=1)
        self.ln(2)

    def add_table(self, headers: list, data: list):
        """Add a simple table."""
        self.set_font("Arial", "B", 10)
        self.set_fill_color(200, 220, 255)

        # Calculate column width
        col_width = (self.w - 20) / len(headers)

        # Headers
        for header in headers:
            self.cell(col_width, 7, header, border=1, fill=True, align="C")
        self.ln()

        # Data
        self.set_font("Arial", "", 10)
        self.set_fill_color(240, 240, 240)
        fill = False
        for row in data:
            for item in row:
                self.cell(col_width, 6, str(item), border=1, fill=fill, align="C")
            self.ln()
            fill = not fill
        self.ln(3)


def parse_markdown_structure(content: str) -> list:
    """
    Parse markdown content into structured elements.
    Returns a list of (type, content) tuples.
    Enhanced to aggressively clean markdown artifacts.
    """
    elements = []
    lines = content.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]

        # Code blocks
        if line.strip().startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if code_lines:
                elements.append(("code", "\n".join(code_lines)))
            i += 1
            continue

        # Headers - ENHANCED with markdown cleaning
        h1_match = re.match(r"^#\s+(.+)$", line)
        h2_match = re.match(r"^##\s+(.+)$", line)
        h3_match = re.match(r"^###\s+(.+)$", line)

        if h1_match:
            # Clean any remaining markdown from header text
            header_text = h1_match.group(1)
            header_text = re.sub(r"\*\*(.+?)\*\*", r"\1", header_text)  # Bold
            header_text = re.sub(r"\*(.+?)\*", r"\1", header_text)  # Italic
            header_text = re.sub(r"__(.+?)__", r"\1", header_text)  # Bold alt
            header_text = re.sub(r"_(.+?)_", r"\1", header_text)  # Italic alt
            header_text = re.sub(r"`(.+?)`", r'"\1"', header_text)  # Inline code
            elements.append(("h1", header_text))
        elif h2_match:
            header_text = h2_match.group(1)
            header_text = re.sub(r"\*\*(.+?)\*\*", r"\1", header_text)
            header_text = re.sub(r"\*(.+?)\*", r"\1", header_text)
            header_text = re.sub(r"__(.+?)__", r"\1", header_text)
            header_text = re.sub(r"_(.+?)_", r"\1", header_text)
            header_text = re.sub(r"`(.+?)`", r'"\1"', header_text)
            elements.append(("h2", header_text))
        elif h3_match:
            header_text = h3_match.group(1)
            header_text = re.sub(r"\*\*(.+?)\*\*", r"\1", header_text)
            header_text = re.sub(r"\*(.+?)\*", r"\1", header_text)
            header_text = re.sub(r"__(.+?)__", r"\1", header_text)
            header_text = re.sub(r"_(.+?)_", r"\1", header_text)
            header_text = re.sub(r"`(.+?)`", r'"\1"', header_text)
            elements.append(("h3", header_text))

        # Bullet lists - ENHANCED with markdown cleaning
        elif re.match(r"^[\s]*[-*+]\s+", line):
            list_items = []
            while i < len(lines) and re.match(r"^[\s]*[-*+]\s+", lines[i]):
                item = re.sub(r"^[\s]*[-*+]\s+", "", lines[i])
                # Clean markdown from list items
                item = re.sub(r"\*\*(.+?)\*\*", r"\1", item)  # Bold
                item = re.sub(
                    r"\*(.+?)\*", r"\1", item
                )  # Italic (after bold to avoid conflicts)
                item = re.sub(r"__(.+?)__", r"\1", item)  # Bold alt
                item = re.sub(r"_(.+?)_", r"\1", item)  # Italic alt
                item = re.sub(r"`(.+?)`", r'"\1"', item)  # Inline code
                item = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1", item)  # Links
                list_items.append(item)
                i += 1
            elements.append(("bullet_list", list_items))
            continue

        # Numbered lists - ENHANCED with markdown cleaning
        elif re.match(r"^[\s]*\d+\.\s+", line):
            list_items = []
            while i < len(lines) and re.match(r"^[\s]*\d+\.\s+", lines[i]):
                item = re.sub(r"^[\s]*\d+\.\s+", "", lines[i])
                # Clean markdown from numbered items
                item = re.sub(r"\*\*(.+?)\*\*", r"\1", item)  # Bold
                item = re.sub(r"\*(.+?)\*", r"\1", item)  # Italic
                item = re.sub(r"__(.+?)__", r"\1", item)  # Bold alt
                item = re.sub(r"_(.+?)_", r"\1", item)  # Italic alt
                item = re.sub(r"`(.+?)`", r'"\1"', item)  # Inline code
                item = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1", item)  # Links
                list_items.append(item)
                i += 1
            elements.append(("numbered_list", list_items))
            continue

        # Blockquotes / Info boxes
        elif line.strip().startswith(">"):
            quote_lines = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote_lines.append(lines[i].strip()[1:].strip())
                i += 1
            if quote_lines:
                elements.append(("info_box", " ".join(quote_lines)))
            continue

        # Regular text - ENHANCED with aggressive markdown cleaning
        elif line.strip():
            # Process inline markdown aggressively
            text = line.strip()

            # Remove horizontal rules
            if re.match(r"^[-*_]{3,}$", text):
                i += 1
                continue

            # Bold (do bold first, before italic to avoid conflicts)
            text = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", text)  # Bold+Italic
            text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # Bold
            text = re.sub(r"__(.+?)__", r"\1", text)  # Bold alt

            # Italic (after bold)
            text = re.sub(r"\*(.+?)\*", r"\1", text)  # Italic
            text = re.sub(r"_(.+?)_", r"\1", text)  # Italic alt

            # Inline code
            text = re.sub(r"`(.+?)`", r'"\1"', text)

            # Links - keep text only, optionally keep URL in parentheses
            text = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1", text)

            # Images - remove completely
            text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

            # Strikethrough
            text = re.sub(r"~~(.+?)~~", r"\1", text)

            # Escape sequences for common markdown artifacts
            text = text.replace("\\*", "*").replace("\\#", "#").replace("\\_", "_")

            # Only add if there's actual content after cleaning
            if text.strip():
                elements.append(("text", text))

        # Empty line
        elif not line.strip():
            elements.append(("blank", ""))

        i += 1

    return elements


def clean_markdown(content: str) -> str:
    """
    Converts Markdown content into clean, readable plain text.
    (Legacy function for backward compatibility)
    """
    # Remove code blocks
    content = re.sub(r"```.*?```", "", content, flags=re.DOTALL)

    # Convert headings
    content = re.sub(
        r"^#{1,6}\s*(.*?)$",
        lambda m: m.group(1).strip().upper(),
        content,
        flags=re.MULTILINE,
    )

    # Remove bold and italic
    content = re.sub(r"(\*\*|__)(.*?)\1", r"\2", content)
    content = re.sub(r"(\*|_)(.*?)\1", r"\2", content)

    # Convert inline code to quotes
    content = re.sub(r"`(.+?)`", r'"\1"', content)

    # Remove images
    content = re.sub(r"!\[.*?\]\(.*?\)", "", content)

    # Convert Markdown links
    content = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1 (\2)", content)

    # Normalize lists
    content = re.sub(r"^\s*[-*+]\s+", "- ", content, flags=re.MULTILINE)

    # Remove blockquotes
    content = re.sub(r"^\s*>\s?", "", content, flags=re.MULTILINE)

    # Remove horizontal rules
    content = re.sub(r"^[-*_]{3,}$", "", content, flags=re.MULTILINE)

    return content.strip()


def save_pdf(
    content: str,
    filename: str = "report.pdf",
    title: str = "Research Report",
    use_structure: bool = True,
    output_dir: str = "generated_reports",
) -> str:
    """
    Converts Markdown content to a beautifully formatted PDF file.

    Args:
        content (str): The Markdown-formatted content to convert.
        filename (str): Desired name of the PDF file.
        title (str): Title of the report for header.
        use_structure (bool): If True, parse markdown structure for better formatting.
                             If False, use simple text cleaning (legacy mode).
        output_dir (str): Directory to save the PDF file.

    Returns:
        str: Path to the generated PDF.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)

    # Create PDF instance
    pdf = EnhancedPDF(title=title)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=25)

    if use_structure:
        # Parse markdown structure for rich formatting
        elements = parse_markdown_structure(content)

        for elem_type, elem_content in elements:
            if elem_type == "h1":
                pdf.chapter_title(elem_content)
            elif elem_type == "h2":
                pdf.section_title(elem_content)
            elif elem_type == "h3":
                pdf.subsection_title(elem_content)
            elif elem_type == "text":
                pdf.body_text(elem_content)
            elif elem_type == "bullet_list":
                pdf.bullet_list(elem_content)
            elif elem_type == "numbered_list":
                pdf.numbered_list(elem_content)
            elif elem_type == "code":
                pdf.code_block(elem_content)
            elif elem_type == "info_box":
                pdf.info_box(elem_content, "info")
            elif elem_type == "blank":
                pdf.ln(3)
    else:
        # Legacy mode: simple text cleaning
        cleaned_content = clean_markdown(content)
        pdf.set_font("Arial", size=11)

        for line in cleaned_content.splitlines():
            line = line.strip()

            if not line:
                pdf.ln(8)
                continue

            if line.isupper() and len(line) <= 60:
                pdf.chapter_title(line.title())
            else:
                pdf.body_text(line)

    pdf.output(file_path)
    return file_path
