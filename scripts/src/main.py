import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# بارگذاری مدل و توکنایزر
model_name = "Qwen/Qwen2.5-32B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# خواندن فایل JSON
path_json = "/home/ubuntu/m15kh/own/LLM/input/pe.json"
with open(path_json, "r", encoding="utf-8") as f:
    json_data = json.load(f)

# آماده‌سازی داده‌ها برای مدل
message = {
    "role": "user",
    "content": f"Analyze this JSON data: {json.dumps(json_data)}"
}

# ارسال پیام به مدل و دریافت پاسخ
inputs = tokenizer([message["content"]], return_tensors="pt", truncation=True, padding=True)
outputs = model.generate(inputs["input_ids"], max_new_tokens=512)

# تبدیل توکن‌ها به متن
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# ذخیره پاسخ در فایل متنی
output_path = "/home/ubuntu/m15kh/own/LLM/output/response.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(response)

print(f"Response saved to {output_path}")
