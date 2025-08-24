import smtplib
import mimetypes
from email.utils import formataddr, make_msgid
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime

def send_helmet_warning_email(
    to_email: str,
    images: list[str] = None,
    *,
    smtp_user: str,
    smtp_password: str,
    smtp_server: str = "smtp.gmail.com",
    smtp_port: int = 587,
    sender_name: str = "Hệ thống Cảnh báo ATGT",
    recipient_name: str | None = None,
    subject: str = "Cảnh báo: Không đội nón bảo hiểm khi tham gia giao thông",
) -> None:
    """
    Gửi email HTML cảnh báo về việc không đội nón bảo hiểm, kèm ảnh minh chứng hiển thị inline.

    Tham số:
        to_email        : email người nhận (ví dụ: "user@example.com")
        images          : danh sách đường dẫn ảnh (png/jpg/...), sẽ hiển thị trong email
        smtp_user       : email đăng nhập SMTP (Gmail), ví dụ "you@gmail.com"
        smtp_password   : App Password của tài khoản Gmail (không dùng mật khẩu thường)
        smtp_server     : máy chủ SMTP (mặc định Gmail)
        smtp_port       : cổng SMTP (587 cho STARTTLS)
        sender_name     : tên hiển thị người gửi
        recipient_name  : tên người nhận, dùng trong lời chào
        subject         : tiêu đề email
    """
    images = images or []

    # Tạo message MIMEMultipart/related để nhúng ảnh
    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"] = formataddr((sender_name, smtp_user))
    msg["To"] = to_email

    # Phần alternative: text/plain và text/html
    alt = MIMEMultipart("alternative")
    msg.attach(alt)

    # ========== Nội dung bản đơn giản (text/plain) – fallback ==========
    plain_text = f"""\
{('Xin chào ' + recipient_name) if recipient_name else 'Xin chào,'}

Đây là thông báo về việc KHÔNG đội nón bảo hiểm khi tham gia giao thông.

• Nguy cơ chấn thương đầu rất cao.
• Ảnh hưởng an toàn của bạn và người xung quanh.
• Hãy đội nón bảo hiểm đạt chuẩn, cài quai đúng cách mọi lúc.

(Email này có phiên bản HTML kèm ảnh minh chứng.)
—
{sender_name} • {datetime.now().strftime('%d/%m/%Y %H:%M')}
"""
    alt.attach(MIMEText(plain_text, "plain", "utf-8"))

    # ========== Mapping ảnh -> Content-ID để nhúng vào HTML ==========
    cid_map = {}
    for img_path in images:
        p = Path(img_path)
        if not p.exists() or not p.is_file():
            continue
        mime_type, _ = mimetypes.guess_type(str(p))
        if not mime_type or not mime_type.startswith("image/"):
            continue

        with open(p, "rb") as f:
            img_data = f.read()

        # Tạo Content-ID duy nhất cho mỗi ảnh
        cid = make_msgid(domain="inline.image").strip("<>")
        cid_map[p.name] = cid

        img_part = MIMEImage(img_data, _subtype=mime_type.split("/")[1])
        img_part.add_header("Content-ID", f"<{cid}>")
        img_part.add_header("Content-Disposition", f'inline; filename="{p.name}"')
        msg.attach(img_part)

    # ========== HTML (gọn, đẹp, dễ đọc, tương thích Gmail) ==========
    # Lưu ý: Dùng style inline + layout đơn giản để tương thích trình mail
    # Ảnh hiển thị theo grid 2 cột (responsive cơ bản).
    def render_image_grid():
        if not cid_map:
            return ""
        cells = []
        for i, (filename, cid) in enumerate(cid_map.items(), start=1):
            cell_html = f"""
            <td style="padding:6px; vertical-align:top; width:50%;">
                <div style="border:1px solid #e5e7eb; border-radius:12px; overflow:hidden;">
                    <img src="cid:{cid}" alt="{filename}" style="display:block; width:100%; height:auto;">
                    <div style="font-size:12px; color:#6b7280; padding:8px 10px; background:#fafafa;">
                        {filename}
                    </div>
                </div>
            </td>
            """
            cells.append(cell_html)

        # Ghép từng hàng 2 cột
        rows = []
        for i in range(0, len(cells), 2):
            pair = cells[i:i+2]
            if len(pair) == 1:
                pair.append('<td style="padding:6px; width:50%;"></td>')
            rows.append(f"<tr>{''.join(pair)}</tr>")
        return f"""
        <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
            {''.join(rows)}
        </table>
        """

    greeting = f"Xin chào {recipient_name}," if recipient_name else "Xin chào,"
    now_str = datetime.now().strftime("%d/%m/%Y %H:%M")

    html = f"""\
<!doctype html>
<html lang="vi">
  <body style="margin:0; padding:0; background:#f5f7fb;">
    <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%" style="background:#f5f7fb;">
      <tr>
        <td align="center" style="padding:24px;">
          <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%" style="max-width:680px; background:#ffffff; border-radius:16px; overflow:hidden; box-shadow:0 4px 14px rgba(0,0,0,0.08);">
            <!-- Header -->
            <tr>
              <td style="background:linear-gradient(135deg,#ef4444,#f59e0b); padding:24px 20px; color:#fff;">
                <div style="font-size:20px; font-weight:700; letter-spacing:0.2px;">⚠️ CẢNH BÁO AN TOÀN GIAO THÔNG</div>
                <div style="font-size:13px; opacity:0.9; margin-top:4px;">Không đội nón bảo hiểm khi tham gia giao thông</div>
              </td>
            </tr>

            <!-- Body -->
            <tr>
              <td style="padding:24px 20px 8px 20px; color:#111827; font-family:Segoe UI, Roboto, Helvetica, Arial, sans-serif; line-height:1.55;">
                <p style="margin:0 0 12px 0; font-size:15px;">{greeting}</p>
                <p style="margin:0 0 14px 0; font-size:15px;">
                  Hệ thống ghi nhận trường hợp <strong>không đội nón bảo hiểm</strong> khi tham gia giao thông.
                  Việc này tiềm ẩn rủi ro an toàn nghiêm trọng. Vui lòng xem thông tin dưới đây:
                </p>

                <div style="border:1px solid #e5e7eb; border-radius:12px; padding:14px; background:#fcfcfc; margin:14px 0;">
                  <ul style="padding-left:18px; margin:0; font-size:14px; color:#374151;">
                    <li>Nguy cơ <strong>chấn thương đầu</strong> tăng cao khi xảy ra va chạm.</li>
                    <li>Ảnh hưởng an toàn cho bản thân và người tham gia giao thông khác.</li>
                    <li>Hãy đội nón bảo hiểm <strong>đạt chuẩn</strong> và cài quai đúng cách mọi lúc.</li>
                  </ul>
                </div>

                <h3 style="font-size:15px; margin:18px 0 10px 0; color:#111827;">Ảnh minh chứng</h3>
                {render_image_grid()}

                <div style="margin-top:16px; padding:14px; background:#f9fafb; border:1px dashed #d1d5db; border-radius:12px; font-size:13px; color:#374151;">
                  Nếu bạn cho rằng có nhầm lẫn, vui lòng phản hồi email này để được hỗ trợ kiểm tra lại.
                </div>

                <p style="margin:16px 0 0 0; font-size:12px; color:#6b7280;">Gửi lúc {now_str}</p>
              </td>
            </tr>

            <!-- Footer -->
            <tr>
              <td style="padding:14px 20px 18px 20px; background:#ffffff; border-top:1px solid #f3f4f6; color:#6b7280; font-size:12px;">
                {sender_name} • Email tự động – vui lòng không chỉnh sửa tiêu đề khi phản hồi
              </td>
            </tr>

          </table>
        </td>
      </tr>
    </table>
  </body>
</html>
"""
    alt.attach(MIMEText(html, "html", "utf-8"))

    # ========== Gửi email ==========
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.ehlo()
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)



