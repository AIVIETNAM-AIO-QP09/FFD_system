# app.py
import streamlit as st
# Nhập hàm giao diện từ file src/predict_ui.py
from src.predict_ui import render_predict_page

# 1. CẤU HÌNH TRANG WEB (Bắt buộc layout="wide" để bổ cục 3 cột ở predict_ui hiển thị đẹp nhất)
st.set_page_config(
    page_title="CarbonOps - Hệ Thống Giám Sát Giao Dịch",
    page_icon="🛡️",
    layout="wide",  # Mở rộng không gian hiển thị sang hai bên màn hình
    initial_sidebar_state="expanded"
)

# 2. GIẢ LẬP TẢI MÔ HÌNH MACHINE LEARNING (BACKEND)
@st.cache_resource
def load_fraud_model():
    """
    Hàm load mô hình AI (XGBoost, CatBoost hoặc ANN).
    Hiện tại trả về None để giao diện chạy thử nghiệm mượt mà.
    Khi bạn có file model.pkl, chỉ cần dùng joblib.load() ở đây.
    """
    try:
        # import joblib
        # model = joblib.load("models/fraud_model.pkl")
        return None
    except Exception:
        return None

# Khởi tạo mô hình ngầm để truyền qua các trang ui
model = load_fraud_model()

# 3. XÂY DỰNG THANH ĐIỀU HƯỚNG (SIDEBAR) TRÊN GIAO DIỆN
with st.sidebar:
    # 🌟 ĐÃ SỬA: Thay use_column_width=True bằng use_container_width=True để xóa Cảnh báo
    st.image(
        "https://images.unsplash.com/photo-1621416894569-0f39ed31d247?w=500&auto=format&fit=crop&q=60", 
        caption="Security Management",
        use_container_width=True 
    )
    st.markdown("### 🎛️ Menu Chức Năng")
    
    # Tạo menu chuyển đổi giữa các tính năng chính của hệ thống
    app_mode = st.sidebar.selectbox(
        "Lựa chọn trang hiển thị:",
        ["Khởi tạo Giao dịch", "Phân tích & Thống kê Tài chính", "Tổng quan Hệ thống"]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Trạng thái Core")
    st.success("● Kênh Core-Banking: Hoạt động")
    st.success("● Pipeline AI Real-time: Sẵn sàng")
    st.caption("Phiên bản cập nhật thử nghiệm năm 2026.")

# 4. ĐIỀU HƯỚNG CÁC TRANG THEO LỰA CHỌN CỦA NGƯỜI DÙNG
if app_mode == "Khởi tạo Giao dịch":
    # Gọi giao diện 3 cột đã code trong src/predict_ui.py (Vẫn giữ nguyên logic bẫy vị trí Tokyo của bạn)
    render_predict_page(model=model)

elif app_mode == "Phân tích & Thống kê Tài chính":
    st.title("📈 Trung Tâm Phân Tích & Giám Sát Tài Chính")
    st.info("💡 **Tính năng đang được phát triển:** Hệ thống trực quan hóa luồng dữ liệu 5 triệu giao dịch và phân tích hành vi rủi ro sẽ được tích hợp tại đây.")
    
    # Thống kê nhanh để trang không bị trống nhìn chuyên nghiệp hơn khi demo ban đầu
    col_stat1, col_stat2 = st.columns(2)
    col_stat1.metric(label="Tổng số mẫu dữ liệu hệ thống", value="5,000,000")
    col_stat2.metric(label="Độ chính xác mô hình dự kiến (Accuracy)", value="99.45 %")

elif app_mode == "Tổng quan Hệ thống":
    st.title("🛡️ Tổng quan Dự án Phát hiện Gian lận Tài chính")
    st.markdown("Hệ thống ứng dụng Machine Learning để phân tích siêu dữ liệu (Metadata) và điểm số hành vi nhằm ngăn chặn giao dịch bất hợp pháp.")
    
    st.subheader("💡 Hướng dẫn vận hành nhanh để Demo")
    st.markdown(
        """
        1. Bấm vào menu bên trái chọn **Khởi tạo Giao dịch**.
        2. Điền thông tin chuyển khoản (Bạn có thể chọn đơn vị tiền là **VND** hoặc **USD**).
        3. Chuyển qua tab **Cấu hình thông số ngầm (Demo)**, thử tăng chỉ số `velocity_score` hoặc `geo_anomaly_score` lên mức cao.
        4. Bấm **CHUYỂN TIỀN NGAY** để xem cơ chế phân tích dữ liệu và phản ứng chặn giao dịch thời gian thực của hệ thống.
        """
    )