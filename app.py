import streamlit as st
import pandas as pd
from OMZ import TWGRILSOptimizer  # ดึง Class ที่คุณเขียนไว้มาใช้

st.set_page_config(page_title="Route Optimizer", layout="wide")

st.title("🤖 Multi-Day Route Optimizer (TW-GRILS)")
st.write("ระบบวางแผนเส้นทางอัจฉริยะ รองรับการอัปโหลดไฟล์ Excel")

# --- ส่วน Sidebar สำหรับรับค่า Input ---
with st.sidebar:
    st.header("⚙️ ตั้งค่าการคำนวณ")
    
    # 1. รับไฟล์ Excel
    uploaded_file = st.file_uploader("อัปโหลดไฟล์ข้อมูล (Excel)", type=["xlsx"])
    
    # 2. รับจำนวนวัน
    num_days_input = st.number_input("จำนวนวันที่ต้องการเดินทาง (NUM_DAYS)", min_value=1, max_value=5, value=2)
    
    # 3. ปุ่มกดรัน
    run_btn = st.button("🚀 เริ่มคำนวณ")

# --- ส่วนการประมวลผล ---
if uploaded_file is not None and run_btn:
    with st.spinner('กำลังประมวลผลด้วย TW-GRILS...'):
        try:
            # สร้าง Instance จาก Class ของคุณ (ส่งไฟล์ที่อัปโหลดเข้าไปตรงๆ ได้เลย)
            opt = TWGRILSOptimizer(
                excel_path=uploaded_file,
                rng_seed=42,
                min_safety=3,
                min_pathway=3,
                max_walking=2500,
                max_total_budget=1000,
                max_travel_time_minutes=200
            )

            # รัน Algorithm ตามจำนวนวันที่เลือก
            routes, vals, stats, total = opt.optimize_multi_day(num_days_input)

            # --- แสดง TRIP SUMMARY ---
            st.header("📊 สรุปภาพรวมการเดินทาง (Trip Summary)")
            td, ts, tc, tm = total
            
            m1, m2, m3 = st.columns(3)
            m1.metric("ระยะทางรวม (km)", f"{td:.2f}")
            m2.metric("คะแนนความพึงพอใจรวม", f"{ts:.2f}")
            m3.metric("ค่าใช้จ่ายรวม (บาท)", f"{tc:.2f}")

            # --- แสดงรายวัน ---
            st.divider()
            for d, (r, v, s_detail) in enumerate(zip(routes, vals, stats), start=1):
                with st.expander(f"📍 รายละเอียดวันที่ {d}", expanded=True):
                    dist_d, score_d, cost_d, metrics_d = s_detail
                    
                    # ตารางเส้นทาง
                    loc_names = [opt.locations[i] for i in r]
                    st.write(f"**เส้นทาง:** {' ➡️ '.join(map(str, loc_names))}")
                    
                    # ตาราง KPI รายวัน
                    st.table(pd.DataFrame([metrics_d]))

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการรัน: {e}")
else:
    st.info("👈 กรุณาอัปโหลดไฟล์ Excel และเลือกจำนวนวันที่ต้องการที่แถบด้านซ้าย")