# Giải thích chi tiết: Xây dựng AI Agent dùng Học tăng cường (RL) để truy xuất trí nhớ dài hạn

Bài học này hướng dẫn cách xây dựng một AI Agent sử dụng Học tăng cường (Reinforcement Learning - RL) để học cách tìm kiếm và trích xuất những "trí nhớ" (thông tin) liên quan nhất từ một kho lưu trữ dài hạn. Mục đích cuối cùng là cung cấp ngữ cảnh chính xác cho Mô hình Ngôn ngữ Lớn (LLM) để nó có thể trả lời câu hỏi một cách chuẩn xác nhất.

Dưới đây là diễn giải chi tiết từng phần mã nguồn và quy trình trong bài viết:

## 1. Thiết lập môi trường và các hàm nền tảng

Phần đầu tiên của bài học tập trung vào việc chuẩn bị các công cụ và định nghĩa những hàm cơ bản nhất để giao tiếp với AI.

*   **Cài đặt và import thư viện**: Mã nguồn cài đặt các thư viện lõi như `openai` (để gọi API của OpenAI), `gymnasium` (để tạo môi trường học tăng cường), `stable-baselines3` (thư viện cung cấp các thuật toán RL có sẵn, ở đây là PPO), cùng với `numpy`, `pandas`, `scikit-learn` để xử lý dữ liệu và toán học.
*   **Xác thực API Key**: Hệ thống sẽ cố gắng lấy `OPENAI_API_KEY` từ nhiều nguồn khác nhau (như từ Google Colab userdata, từ biến môi trường hệ điều hành, hoặc yêu cầu người dùng nhập trực tiếp) để đảm bảo việc gọi API không bị gián đoạn.
*   **Hàm `embed_texts`**: Văn bản của con người cần được chuyển thành các con số để máy tính hiểu. Hàm này sử dụng mô hình `text-embedding-3-small` của OpenAI để biến đổi danh sách các đoạn văn bản (trí nhớ và câu hỏi) thành các vector ngữ nghĩa (embeddings) và chuẩn hóa độ dài của chúng.
*   **Hàm `chat_answer`**: Đây là bước cuối cùng của hệ thống. Hàm này nhận vào một câu hỏi và các trí nhớ đã được chọn lọc, ghép chúng thành một đoạn prompt (lời nhắc). Nó thiết lập vai trò (system prompt) ép LLM (`gpt-4o-mini`) phải đóng vai trò là một trợ lý hỏi-đáp nghiêm ngặt: chỉ được dùng thông tin trong trí nhớ được cung cấp để trả lời, nếu không có thì phải từ chối trả lời.
*   **Hàm `llm_judge_exact`**: Để tự động hóa việc chấm điểm cho Agent, hàm này dùng LLM đóng vai trò là một "giám khảo" nghiêm khắc. Nó so sánh câu trả lời do hệ thống sinh ra (`predicted_answer`) với câu trả lời chuẩn (`gold_answer`). Kết quả trả về duy nhất ở định dạng JSON chứa điểm `score` (1.0 nếu đúng ý nghĩa, 0.0 nếu sai).

## 2. Xây dựng Kho Trí nhớ (Memory Bank) và Tập Câu hỏi (Queries)

Để Agent có dữ liệu để học tập, chúng ta không dùng dữ liệu thật mà sinh ra một bộ dữ liệu giả lập (synthetic dataset).

*   **Định nghĩa Cấu trúc Trí nhớ (`MemoryItem`)**: Mỗi đoạn trí nhớ được biểu diễn bằng một cấu trúc dữ liệu (`dataclass`) bao gồm: ID của trí nhớ, chủ đề (`topic`), thực thể được nhắc đến (`entity`), thuộc tính của thực thể (`slot`), giá trị thực sự (`value`), và cuối cùng là câu văn bản hoàn chỉnh (`text`).
*   **Tạo Kho Trí nhớ (`build_memory_bank`)**: 
    *   Bài học tạo ra các dữ kiện gốc (facts) về nhiều lĩnh vực như robot (Astra), thiên văn (Orion), y học (Vita)...
    *   **Tạo thông tin hữu ích**: Từ các dữ kiện gốc, mã nguồn dùng các mẫu câu (`phrasing_templates`) để sinh ra nhiều cách diễn đạt khác nhau cho cùng một thông tin (ví dụ: "Robot Astra có thời lượng pin là 18 giờ").
    *   **Tạo thông tin nhiễu (Distractors)**: Đây là một bước rất hay. Hệ thống cố tình sinh ra các câu chứa tên thực thể (ví dụ: "Astra") nhưng lại không có thông tin kỹ thuật nào (như: "Astra đã được nhắc đến trong cuộc họp tóm tắt"). Ngoài ra còn có các câu nhiễu chung chung ("Bảo trì hệ thống vào thứ Ba"). Mục đích là làm cho bài toán khó hơn, mô phỏng đúng thực tế để thử thách khả năng chọn lọc của Agent.
*   **Tạo Câu hỏi (`build_queries`)**: Tương ứng với các thông tin có ích, hệ thống dùng các mẫu câu hỏi ngẫu nhiên để tạo ra câu hỏi. Nó ghi nhớ lại đâu là giá trị chuẩn (`gold_value`) và ID của trí nhớ chứa câu trả lời đúng đó (`gold_memory_id`).
*   Tất cả trí nhớ và câu hỏi đều được đưa qua hàm `embed_texts` để biến thành vector số học.

## 3. Tìm ứng viên (Top-K) và Thiết lập Môi trường Học Tăng Cường (RL Environment)

Không thể để RL Agent duyệt qua hàng ngàn trí nhớ cùng lúc, hệ thống sử dụng một bước lọc sơ bộ trước.

*   **Lọc Top ứng viên (`get_top_k_candidates`)**: Dựa trên phương pháp truyền thống là Độ tương đồng Cosine (Cosine Similarity), hệ thống so sánh vector của câu hỏi với toàn bộ kho trí nhớ để lấy ra 8 câu có vẻ liên quan nhất. 
*   **Tạo Đặc trưng cho Agent (`build_state_features`)**: Với 8 ứng viên này, hệ thống sẽ tính toán các đặc trưng (features) chi tiết hơn để "mớm" cho Agent, bao gồm:
    *   `sim`: Điểm tương đồng ngữ nghĩa bằng vector.
    *   `overlap`: Tỷ lệ từ khóa trùng lặp giữa câu hỏi và câu trí nhớ.
    *   `entity_match` / `slot_match`: Kiểm tra xem tên thực thể hay thuộc tính trong câu hỏi có xuất hiện trong câu trí nhớ dạng chữ hay không.
    *   `rank`: Thứ hạng từ 1 đến 8 do thuật toán tương đồng xếp hạng ban đầu.
    Tất cả các con số này ghép lại thành một Mảng trạng thái (State) đại diện cho câu hỏi hiện tại.
*   **Môi trường `MemoryRetrievalEnv`**: Kế thừa từ thư viện Gym, đây là thế giới ảo để Agent rèn luyện.
    *   `reset()`: Lấy ngẫu nhiên một câu hỏi và hiển thị Mảng trạng thái của 8 ứng viên cho Agent xem.
    *   `step(action)`: Agent quyết định chọn 1 trong 8 ứng viên. 
    *   **Phần thưởng (Reward)**: Nếu Agent chọn đúng trí nhớ chứa câu trả lời (`is_gold`), nó được cộng ngay +2.0 điểm. Nếu trí nhớ đó chứa đúng tên thực thể, được cộng +0.8; đúng thuộc tính cộng +0.6... Ngược lại, nếu chọn ứng viên xếp hạng thấp mà không đúng, nó sẽ bị trừ điểm theo thứ hạng. Chính hàm phần thưởng này là thứ uốn nắn hành vi của Agent.

## 4. Huấn luyện (Training) và Đánh giá (Evaluation) Agent

*   **Chia tập dữ liệu**: Tập dữ liệu câu hỏi được chia thành 70% để học (train), 15% để kiểm tra trong lúc học (validation) và 15% để thi thật (test).
*   **Huấn luyện với thuật toán PPO**: Hệ thống dùng mô hình PPO (`Proximal Policy Optimization` của `stable-baselines3`) để huấn luyện Agent trong 12.000 bước. Trong quá trình này, Agent liên tục nhận câu hỏi, chọn ứng viên, nhận phần thưởng/hình phạt và cập nhật lại bộ não (Mạng nơ-ron đa tầng - MlpPolicy) của mình để tìm ra chiến thuật chọn thông minh nhất.
*   **Đánh giá trực tiếp khả năng Truy xuất**: 
    Hàm `evaluate_retriever` so sánh hai phương pháp:
    *   `Baseline`: Phương pháp truyền thống, nhắm mắt chọn luôn ứng viên có độ tương đồng vector (`sim`) cao nhất.
    *   `RL Agent`: Phương pháp mới, dùng mô hình PPO đã huấn luyện để phân tích đa chiều rồi mới quyết định chọn ai.
    Kết quả đo lường độ chính xác (Accuracy) được thống kê thành bảng `DataFrame` và vẽ biểu đồ thanh (`matplotlib`), giúp trực quan hóa việc RL Agent có thực sự vượt trội hơn Baseline hay không.

## 5. Kiểm tra thực tế với LLM và Demo tương tác

Việc truy xuất thông tin đúng là chưa đủ, mục đích cuối là LLM có trả lời đúng hay không (Downstream QA).

*   **So sánh khả năng trả lời câu hỏi**: Hàm `answer_with_retriever` sẽ lấy trí nhớ do Baseline và RL Agent chọn, đưa riêng từng cái vào hàm `chat_answer` để LLM sinh câu trả lời. Sau đó đưa câu trả lời đó cho Giám khảo LLM (`llm_judge_exact`) chấm điểm. Hệ thống tiếp tục vẽ biểu đồ so sánh chất lượng câu trả lời cuối cùng giữa 2 phương pháp.
*   **Kiểm tra chi tiết (Inspect Examples)**: Hàm `inspect_examples` cho phép người dùng nhìn thấu bên trong. Nó in ra thành bảng cho thấy với câu hỏi này, Baseline đã chọn đoạn text nào (đúng hay sai) và RL Agent đã chọn đoạn text nào. Nó thường cho thấy Baseline hay bị lừa bởi các câu "distractor" (nhiễu), trong khi RL Agent khôn ngoan hơn.
*   **Demo Tương tác (`interactive_demo`)**: Mã nguồn cung cấp một tính năng thú vị để người lập trình tự gõ câu hỏi bất kỳ (ví dụ: "Tell me the country for Cedar."). Hệ thống sẽ in ra quá trình theo thứ tự: Câu hỏi -> Top ứng viên truyền thống -> Trí nhớ mà RL Agent chốt -> Câu trả lời mượt mà do LLM sinh ra.
*   **Lưu trữ dữ liệu (Artifacts)**: Ở cuối script, toàn bộ biểu đồ, kết quả so sánh CSV, vector nhúng, mô hình RL đã huấn luyện và kho trí nhớ JSON được lưu vào một thư mục (`rl_agent_memory_retrieval_artifacts`) để làm bằng chứng hoặc để sử dụng cho các dự án sau này.

## Tổng kết bài học

Bài học đi từ việc tạo dữ liệu, trích xuất đặc trưng, đến huấn luyện Học tăng cường và kết hợp Generative AI. Điểm mấu chốt của bài học là:
Sử dụng tìm kiếm bằng Vector (Semantic Search/Cosine Similarity) rất tốt nhưng dễ bị "đánh lừa" bởi các câu văn dài dòng có cấu trúc tương tự nhưng thiếu thông tin cốt lõi. Bằng cách thiết kế một môi trường Học Tăng Cường với hệ thống phần thưởng hợp lý, chúng ta có thể dạy cho một AI Agent kết hợp cả độ tương đồng vector, sự trùng khớp từ khóa và thực thể để chọn ra mẩu thông tin chính xác tuyệt đối, từ đó giúp Hệ thống Hỏi Đáp bằng LLM trở nên đáng tin cậy hơn rất nhiều.
