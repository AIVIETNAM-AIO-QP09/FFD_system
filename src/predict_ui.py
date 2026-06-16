# src/predict_ui.py
import streamlit as st
import pandas as pd
from datetime import datetime

def render_predict_page(model=None):
    # 1. Các tham số cấu hình ngầm (Backend)
    EXCHANGE_RATE = 26309.84
    
    # 2. CHIA BỐ CỤC 3 CỘT (Tỷ lệ: 20% | 60% | 20%)
    col_left, col_main, col_right = st.columns([1, 3, 1])
    
    # --- CỘT TRÁI (col_left) ---
    with col_left:
        st.markdown("### 🔒 Bảo mật")
        st.image(
            "https://images.unsplash.com/photo-1563986768609-322da13575f3?w=500&auto=format&fit=crop&q=60", 
            caption="Hệ thống bảo mật AI 24/7", 
            use_container_width=True
        )
        st.caption("Mọi giao dịch đều được mã hóa đầu cuối và giám sát bởi mô hình phát hiện gian lận chủ động.")

    # --- CỘT GIỮA (col_main) ---
    # --- CỘT GIỮA (col_main) ---
    with col_main:
        st.title("🛡️ Cổng Giao Dịch An Toàn")
        
        # BƯỚC 1: Xử lý logic gán tiền tệ động trực tiếp vào Session State (Phá cache của Form)
        # Ta tạo một radio chọn tiền tệ nằm NGOÀI Form hoặc xử lý đổi key đồng bộ
        tab_banking, tab_demo_scores = st.tabs(["💳 Giao dịch chuyển tiền", "⚙️ Cấu hình thông số ngầm"])
        
        # Biến hứng data từ tab thông số ngầm
        time_since_last = 1.0
        spending_dev = 0.0
        velocity = 1
        geo_anomaly = 0.0

        with tab_demo_scores:
            st.markdown("##### 🕵️ Các chỉ số phân tích hành vi ngầm của tài khoản")
            st.caption("Kéo chỉnh các thông số này để tạo kịch bản gian lận nhằm kiểm tra phản ứng của mô hình AI.")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                time_since_last = st.number_input("Thời gian từ giao dịch trước (Số giờ)", min_value=0.0, value=1.0, step=0.1)
                spending_dev = st.slider("Mức độ đột biến chi tiêu (spending_deviation_score)", min_value=-5.26, max_value=5.02, value=0.0, step=0.1)
            with col_s2:
                velocity = st.slider("Số lượng giao dịch liên tiếp gần đây (velocity_score)", min_value=1, max_value=20, value=1)
                geo_anomaly = st.slider("Điểm bất thường vị trí GPS (geo_anomaly_score)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

        with tab_banking:
            # BƯỚC 2: Tách ô chọn loại tiền tệ ra khỏi Form (Đặt ngay trước Form) 
            # để khi click chọn VND/USD, trang lập tức ghi nhận và thay đổi số tiền mẫu bên dưới
            currency = st.radio("Loại tiền tệ", ["VND", "USD"], horizontal=True)
            default_amount = 50000.0 if currency == "VND" else 20.0
            
            # BƯỚC 3: Triển khai Form nhập liệu chính
            with st.form("banking_transfer_form"):
                col_sub1, col_sub2 = st.columns(2)
                
                with col_sub1:
                    account_dest = st.text_input("Số tài khoản/Mã ví người nhận", placeholder="Nhập số tài khoản...")
                    tx_type = st.selectbox("Hình thức chuyển tiền", options=["transfer", "payment", "deposit", "withdrawal"])
                    
                with col_sub2:
                    # 🌟 ĐÂY LÀ ĐIỂM CHỐT: Dùng key động kết hợp currency để Streamlit hủy hoàn toàn cache cũ 
                    # và render lại ô nhập số tiền với đúng định dạng mặc định tương ứng!
                    amount_input = st.number_input(
                        f"Số tiền giao dịch ({currency})", 
                        min_value=0.01, 
                        value=default_amount, 
                        key=f"amount_input_field_{currency}" 
                    )
                    
                st.markdown("---")
                col_sub3, col_sub4 = st.columns(2)
                with col_sub3:
                    ui_location = st.selectbox("Vị trí hiện tại", ["Việt Nam", "Tokyo", "New York", "Singapore", "Berlin", "Sydney", "Toronto", "Dubai", "London"])
                with col_sub4:
                    device = st.selectbox("Thiết bị đăng nhập", options=["mobile", "web", "atm", "pos"])
                    
                submitted = st.form_submit_button("🚀 CHUYỂN TIỀN NGAY")
    # --- CỘT PHẢI (col_right) ---
    with col_right:
        st.markdown("### 📢 Tin tức") 
        st.image(
            "https://images.unsplash.com/photo-1559526324-4b87b5e36e44?w=500&auto=format&fit=crop&q=60", 
            caption="Ứng dụng Ngân hàng số thế hệ mới", 
            use_container_width=True
        )
        st.info("💡 **Mẹo an toàn:** Tuyệt đối không cung cấp mã OTP hoặc nhấn vào các đường link lạ gửi qua SMS.")

    # 3. LUỒNG XỬ LÝ LOGIC (Nằm trong col_main để hiển thị kết quả ngay dưới Form)
    with col_main:
        if submitted:
            if not account_dest:
                st.warning("⚠️ Vui lòng nhập số tài khoản người nhận trước khi thực hiện.")
                return

            # Xử lý quy đổi ngầm tiền tệ
            amount_in_usd = amount_input
            if currency == "VND":
                amount_in_usd = amount_input / EXCHANGE_RATE
                
            if amount_in_usd < 0.01 or amount_in_usd > 3520.57:
                st.error(f"❌ Giao dịch thất bại. Số tiền vượt quá hạn mức xử lý tối đa của hệ thống (Tối đa $3,520 USD).")
                return

            # LOGIC DEMO: Bẫy chuyển vùng Việt Nam -> Tokyo
            final_location = "Tokyo" if ui_location == "Việt Nam" else ui_location
            user_ip = "192.168.1.45"

            # Đóng gói vector dữ liệu
            feature_dict = {
                "amount": amount_in_usd,
                "type": tx_type,
                "device_used": device,
                "location": final_location,
                "ip_address": user_ip,
                "time_since_last_transaction": time_since_last,
                "spending_deviation_score": spending_dev,
                "velocity_score": velocity,
                "geo_anomaly_score": geo_anomaly,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            input_df = pd.DataFrame([feature_dict])

            st.markdown("---")
            with st.spinner("🔄 Hệ thống đang xác thực an toàn giao dịch..."):
                # Luật giả lập đánh giá rủi ro
                if geo_anomaly > 0.8 or velocity > 15:
                    st.error("❌ **GIAO DỊCH BỊ TỪ CHỐI:** Hệ thống phát hiện dấu hiệu chiếm đoạt tài khoản hoặc định vị bất hợp pháp.")
                else:
                    st.success(f"✅ **GIAO DỊCH THÀNH CÔNG!** Đã chuyển {amount_input:,.2f} {currency} đến tài khoản {account_dest}.")
            
            with st.expander("🔍 Chi tiết kỹ thuật (Xem Log Vector đầu vào)"):
                st.dataframe(input_df)