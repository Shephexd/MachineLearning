import smtplib


def mail_alert(ID, PWD, FROM, TO, msg=""):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(ID, PWD)
    server.sendmail(FROM, TO, msg)
    server.quit()
