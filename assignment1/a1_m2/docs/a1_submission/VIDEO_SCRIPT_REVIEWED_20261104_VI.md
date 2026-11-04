# CO3133-Semester-261 – Group G-M10 – Assignment 1

**Trần Gia Lâm — 2352670 · Nguyễn Hữu Cầu — 2352129**

Bài trình bày gồm nghiên cứu năm mô hình Fashion-MNIST và phần mở rộng phân loại trang phục RGB.


## Slide 1. Bài toán và câu hỏi nghiên cứu — Lâm

Dạ, em chào thầy và các bạn. Em là Trần Gia Lâm, trưởng nhóm G-M10. Cùng thực hiện với em là bạn Nguyễn Hữu Cầu. Đây là Assignment 1 môn CO3133, lớp thầy Lê Thành Sách, học kỳ 261. Trong nhóm, em phụ trách điều phối, quy trình thí nghiệm, chạy các lượt huấn luyện trên Colab A100, tổng hợp và phân tích kết quả, tích hợp báo cáo. Bạn Cầu phụ trách dữ liệu, EDA, MLP và các phần phân tích kỹ thuật liên quan.

Nhóm em so sánh năm họ mô hình trên Fashion-MNIST: Linear, MLP, CNN tự thiết kế, GRU và Transformer. Câu hỏi nhóm em đặt ra là: cùng một bài toán, nếu thay cách biểu diễn ảnh và cách huấn luyện, thì chất lượng, kiểu lỗi và chi phí sẽ thay đổi như thế nào?

Bài trình bày có ba phần. Phần đầu là phép so sánh trên official test, đã chốt từ trước. Phần hai là vòng tối ưu hyperparameter, chọn cấu hình bằng validation và không chấm lại test. Hai phần này có vai trò khác nhau nên em sẽ báo cáo tách riêng. Phần cuối, nhóm em mở rộng sang phân loại trang phục trên ảnh RGB thật.

Trước hết, em mời bạn Cầu trình bày dữ liệu và cách chia tập.

## Slide 2. Dữ liệu Fashion-MNIST — Cầu

Em là Nguyễn Hữu Cầu, phụ trách dữ liệu, EDA và phần MLP. Em xin trình bày dữ liệu và cách chia tập trước.
Mỗi mẫu dữ liệu là một ảnh xám 28×28, và model phải đoán ảnh đó thuộc class nào trong mười class quần áo. Official training có 60.000 ảnh, official test có 10.000 ảnh. Nhóm em chia phần training thành 54.000 ảnh train và 6.000 ảnh validation, stratified theo nhãn, seed 42.

Tính ra mỗi class có 5.400 ảnh train, 600 ảnh validation và 1.000 ảnh test, tức là nhãn cân bằng hoàn toàn. Nhưng cân bằng không có nghĩa là các class khó như nhau. Nhìn các ảnh mẫu, thầy và các bạn sẽ thấy mấy loại áo có đường nét rất gần nhau ở kích thước nhỏ như vậy. Còn model thực sự nhầm nhiều tới đâu thì nhóm em đo bằng predictions và confusion matrix, em sẽ nói kỹ ở phần error analysis.


## Slide 3. Vai trò của các tập dữ liệu — Cầu

Trong vòng tối ưu, nhóm em chia tiếp 54.000 ảnh train thành 48.600 ảnh inner-train và 5.400 ảnh inner-validation, với split seed cố định. Giai đoạn discovery và confirmation chỉ dùng hai tập này.

Chọn xong recipe cho từng family, nhóm em train lại từ đầu trên đủ 54.000 ảnh, với ba seed 42, 43 và 44. Ở bước final này, tập validation 6.000 ảnh ban đầu được dùng để chọn checkpoint cuối.

Có một điểm em muốn nói rõ ngay từ đầu: official test đã được xem ở lần so sánh trước. Vì vậy vòng tối ưu này không chấm lại trên test, và các con số mới là kết quả validation mang tính thăm dò, không phải điểm test độc lập. Nhóm em có kiểm tra index và fingerprint để split giữ nhất quán qua các lần chạy, nhưng cách này không thay thế được việc kiểm tra ảnh trùng nội dung.

Sau đây, em mời bạn Lâm trình bày pipeline huấn luyện.

## Slide 4. Pipeline và training loop — Lâm

Pipeline khá đơn giản. ToTensor đưa pixel từ 0–255 về float32 trong khoảng 0–1. Mỗi batch có shape B×1×28×28, label là int64 với shape B. Preprocessing giống nhau cho mọi model, và vòng này nhóm em không dùng augmentation.

Model nào cũng trả ra mười logits. Logits được đưa thẳng vào CrossEntropyLoss, không qua softmax trước, vì CrossEntropyLoss đã có sẵn log-softmax bên trong. Mỗi bước train gồm xóa gradient, forward, backward, gradient clipping rồi optimizer step. Khi validation, model chuyển sang eval mode, tắt gradient và không cập nhật weights.

Loss được cộng theo số mẫu, nên batch cuối dù ngắn hơn vẫn được tính đúng trọng số. Accuracy và Macro-F1 được tính trên toàn bộ predictions của cả split, chứ không lấy trung bình theo từng batch.


## Slide 5. Năm kiến trúc — Cầu và Lâm

**Cầu:** Linear đơn giản nhất: flatten 784 pixel rồi đưa qua một linear layer. MLP thêm một hidden layer 256 chiều với ReLU để học quan hệ phi tuyến giữa các pixel. Hai model cùng dùng vector ảnh làm đầu vào; MLP có 203.530 tham số. Trong đó, em phụ trách phần MLP.

**Lâm:** Ba kiến trúc còn lại khai thác ảnh theo cách khác. CNN tự thiết kế dùng convolution 32, 64, 128 channel, pooling và classifier, không dùng pretrained backbone.

GRU coi mỗi hàng 28 pixel là một timestep, tổng cộng 28 timestep, hidden size 128. Transformer chia ảnh thành 49 patch 4×4, project thành embedding 128 chiều, cộng positional embedding, rồi qua hai encoder layer với bốn attention head.

Như vậy, nhóm so sánh vector pixel, feature map, chuỗi hàng và patch token. Số tham số giữa các family không bằng nhau, nên khi đọc kết quả phải xét cả capacity.

## Slide 6. Phép so sánh official test gốc — Lâm

Đây là kết quả trên official test. Ở lần chạy gốc, mỗi model train 20 epoch với seed 42. Checkpoint được chọn theo validation loss thấp nhất và chốt lại trước khi chạy test.

Test accuracy lần lượt là: Linear 84,68%, MLP 88,74%, CNN 89,59%, GRU 89,14% và Transformer 88,94%. Trong lần so sánh này, CNN cao nhất, hơn Linear 4,91 điểm phần trăm.

Bảng này nhóm em giữ nguyên. Các con số của vòng HPO phía sau, nhóm em không gán cho official test, vì test đã xem rồi thì có tối ưu tiếp cũng không tạo ra được một test set mới.


## Slide 7. Thiết kế tìm kiếm recipe — Lâm

Tiếp theo là cách nhóm em tìm recipe. Cả study gồm 61 lượt chạy, mỗi lượt 60 epoch: 36 lượt discovery, 10 lượt confirmation và 15 lượt final. Linear có bốn recipe, còn mỗi family phi tuyến có tám recipe.

Trong mỗi family, các recipe gồm một anchor, vài controlled probe và một số cấu hình lấy mẫu bằng Optuna RandomSampler. Nhóm em không dùng Bayesian optimization, cũng không pruning theo score. Những gì được tìm là learning rate, weight decay, dropout và loại schedule, còn kích thước kiến trúc giữ cố định.

Về thời gian, nhóm em đặt trần bảy ngày cho controller, nhưng đó chỉ là giới hạn trên. Thực tế controller ghi nhận cả study chạy xong trong khoảng 8 tiếng 20 phút.


## Slide 8. Confirmation và chọn recipe — Lâm

Recipe được chọn theo hai bước. Ở discovery, nhóm em chạy seed 101 để tìm challenger tốt nhất, tức recipe tốt nhất ngoài anchor. Sau đó chạy thêm anchor và challenger với seed 202, rồi so Macro-F1 trung bình trên hai seed 101 và 202. Nếu chênh lệch không quá 1e-6 thì giữ anchor.

Chỗ này có một điểm yếu nhóm em muốn nói thẳng: seed 101 đã tham gia vào việc chọn challenger, nên chỉ seed 202 là lần xác nhận thật sự mới. Transformer là ví dụ rõ nhất. Ở seed 101, C1 tăng hơn nửa điểm phần trăm F1, nhưng sang seed 202 lại giảm khoảng 0,2 điểm. Tính trung bình thì C1 vẫn được chọn, nhưng chưa đủ bằng chứng để nói là cải thiện ổn định.


## Slide 9. Các recipe được chọn — Lâm

Kết quả chọn recipe: Linear, GRU và Transformer chọn C1; MLP chọn C7; CNN chọn C5. Linear và GRU dùng cosine schedule, còn Transformer warmup năm epoch rồi mới cosine.

MLP C7 có learning rate khoảng 0,001, weight decay 1e-5 và head dropout 0,2. CNN C5 có learning rate khoảng 0,0014, weight decay 1e-4, dropout 0,3 và giữ learning rate constant.

CNN chọn constant, còn các family khác chọn schedule. Trong vòng tìm kiếm này, không có một recipe chung tốt nhất cho mọi model. Và vì mỗi recipe thay đổi nhiều yếu tố cùng lúc, nhóm em không quy chênh lệch cho riêng dropout hay riêng learning rate.


## Slide 10. Kết quả validation qua ba seed — Lâm

Chốt recipe xong, nhóm em train lại mỗi family với ba seed 42, 43 và 44. Bảng báo cáo đủ từng seed, giá trị trung bình và sample standard deviation, chứ không chọn seed đẹp nhất để đại diện cho family.

CNN C5 dẫn đầu: validation accuracy trung bình khoảng 93,28%, độ lệch chuẩn chỉ khoảng 0,06 điểm phần trăm. Macro-F1 trung bình khoảng 0,933, cũng cao nhất trong các recipe final.

MLP, Transformer và GRU bám nhau rất sát, Macro-F1 đều quanh 0,914. Khoảng cách nhỏ như vậy thì phải đọc cùng độ dao động giữa các seed. Nhóm em chỉ kết luận trong phạm vi này: ba seed trên một tập validation đã dùng trước đó chưa đủ để khẳng định model nào vượt trội nói chung.


## Slide 11. Checkpoint tốt nhất khác epoch cuối — Lâm

Slide này cho thấy checkpoint tốt nhất không nhất thiết là epoch cuối. Ở vòng này, nhóm em chọn checkpoint theo Macro-F1 cao nhất; nếu hòa thì lấy CE thấp hơn, hòa nữa thì lấy epoch sớm hơn. Quy tắc này khác với lần chạy gốc, khi đó chọn theo CE thấp nhất.

Ví dụ với CNN C5 seed 42: checkpoint theo F1 rơi vào epoch 44, F1 khoảng 0,933. Checkpoint có CE thấp nhất lại ở epoch 51. Còn tới epoch 60, train accuracy vẫn rất cao nhưng validation F1 xuống còn khoảng 0,926.

Vì vậy nhóm em không mặc định lấy epoch cuối, và cũng không ghép kiểu lấy F1 của epoch 44 với loss đẹp nhất của epoch 51. Mỗi dòng trong bảng kết quả phải ứng với đúng một checkpoint có thật.

Tiếp theo, bạn Cầu sẽ trình bày phân tích lỗi và chi phí.

## Slide 12. Error analysis — Cầu

Giờ mình xem model sai ở đâu. Với CNN C5 seed 42, trên 6.000 ảnh validation, các lỗi nổi bật là: 49 ảnh Shirt bị đoán thành T-shirt/top, 45 ảnh T-shirt/top bị đoán thành Shirt, và 42 ảnh Coat bị đoán thành Shirt. Mỗi class có 600 ảnh, nên tỷ lệ tương ứng khoảng 8,2%, 7,5% và 7%.

Tức là các loại áo vẫn khó phân biệt. Hình dáng gần giống nhau và độ phân giải thấp là những giả thuyết hợp lý, nhưng đó là giả thuyết để đem đi đối chiếu với ảnh. Chỉ nhìn confusion matrix thì chưa chứng minh được nguyên nhân.

Các ví dụ đúng, sai trên slide chỉ để minh họa; muốn biết lỗi phổ biến tới đâu thì phải nhìn tần suất trên toàn tập. Và các con số này là của riêng seed 42, không phải confusion matrix trung bình ba seed.


## Slide 13. Chi phí và fairness — Cầu

Về chi phí, CNN chỉ có khoảng 94 nghìn tham số, ít hơn nhiều so với MLP khoảng 204 nghìn và Transformer khoảng 275 nghìn. Ở các lượt final, fit time trung bình của CNN khoảng 482 giây, Transformer khoảng 680 giây, trên cùng loại A100. Fit time ở đây gồm train và validation của các epoch đã chạy xong, không tính thời gian chạy cả notebook hay tổng thời gian được cấp GPU.

Inference được đo riêng phần forward, batch 128, input đã nằm sẵn trên GPU, có warm-up và CUDA synchronization. Con số này không gồm DataLoader hay thời gian chuyển dữ liệu, nên cũng không phải latency khi xử lý từng ảnh một.

Về fairness, nhóm em giữ cùng split và cùng số epoch, nhưng recipe, capacity và số candidate thì khác nhau. Nên đây không phải phép so sánh ngang bằng về số tham số hay tổng compute.

Em xin chuyển lại cho bạn Lâm để tổng kết phần Fashion-MNIST và trình bày phần mở rộng.

## Slide 14. Kết luận của nghiên cứu OPT3 — Lâm

Tóm lại phần OPT3. Trong các cấu hình đã đo, CNN C5 có Macro-F1 validation trung bình cao nhất và dao động nhỏ qua ba seed. Linear vẫn là baseline đơn giản. GRU và Transformer cho thấy ảnh hoàn toàn có thể được xử lý như chuỗi hàng hoặc chuỗi patch token.

Nhưng kết quả này chưa đủ để nói HPO là lý do duy nhất giúp điểm cao hơn lần chạy cũ. Ở OPT3, nhóm em đã đổi cả tiêu chí chọn checkpoint lẫn protocol, và lúc đó chưa có lượt final của anchor C0 để đối chiếu. Hạn chế này được xử lý ở vòng đối chứng bổ sung, em sẽ trình bày ở slide 23.

Ngoài ra, tập validation final đã được dùng từ trước, còn official test gốc vẫn để riêng. Vì vậy kết luận của nhóm em chỉ nằm trong phạm vi benchmark, các recipe và các seed đã báo cáo.


## Slide 15. Tái lập và demo — Lâm

Để có thể kiểm tra lại, nhóm em lưu đầy đủ source snapshot, candidate manifest, split ID, task binding, lịch sử đủ 60 epoch, hai checkpoint theo F1 và theo CE, cùng toàn bộ predictions. Từ predictions, số liệu được tính lại và đối chiếu với nhãn và sample ID; bước selection và phép tổng hợp ba seed cũng được kiểm tra lại.

Ví dụ với CNN C5 seed 42, checkpoint Macro-F1 được chọn ở epoch 44. File history.csv và file predictions gắn kết quả đó với đúng epoch, đúng sample ID và nhãn thật, nên ai cũng có thể tự đối chiếu độc lập từ dữ liệu đã lưu.

Các study huấn luyện là em tự chạy trên Google Colab A100. Đợt RGB bổ sung và các phép đối chiếu độc lập thì chạy local trên CPU.


## Slide 16. GarmentLab: thí nghiệm RGB v1 — Lâm

Sau phần Fashion-MNIST, nhóm em mở rộng sang ảnh RGB thật: phân loại sáu loại trang phục trên ảnh Fashionpedia. Lần này nhóm em dùng MobileNetV3-Small pretrained trên ImageNet. Phần này bổ sung cho năm model tự train của A1, chứ không thay thế.

Dữ liệu gồm 17.827 crop. Nhóm em tách riêng bốn tập fit, selection, calibration và assessment, đồng thời kiểm tra source group, ảnh trùng và các nhóm ảnh gần trùng giữa các tập. Cuối cùng assessment còn 327 crop từ 221 source group. Recipe được chọn theo Macro-F1 trung bình trên selection, còn seed dùng để triển khai là 314, đã chốt trước khi đánh giá trên assessment.

Hai recipe frozen và fine-tune đều chạy ba seed, mỗi lượt 18 epoch. Trên assessment, frozen đạt trung bình khoảng 74,4% accuracy và Macro-F1 0,677. Fine-tune cao hơn khá nhiều: khoảng 82,1% accuracy và Macro-F1 0,775. Riêng seed 314 đạt 83,8% và 0,790. Vì ba seed dùng chung một split, độ lệch giữa chúng phản ánh dao động do huấn luyện.


## Slide 17. Confidence và review policy của v1 — Lâm

Fine-tune giúp phân loại tốt hơn, nhưng xác suất đầu ra lại quá tự tin. Nên với seed 314, nhóm em fit temperature trên tập calibration, được khoảng 4,92, rồi giữ cố định. Trên assessment, NLL giảm từ khoảng 1,36 xuống 0,48, ECE giảm từ khoảng 0,134 xuống 0,062. Temperature scaling không làm đổi nhãn dự đoán, nên accuracy giữ nguyên.

Ngưỡng confidence được chọn trên calibration, với mục tiêu accepted accuracy 95%. Sang assessment, model chấp nhận 228 trên 327 crop, đúng 211 crop, tức accepted accuracy khoảng 92,5% tại coverage gần 70%. Nghĩa là con số 95% đặt ra lúc chọn ngưỡng không phải là bảo đảm khi gặp ảnh mới, và nhóm em không trình bày nó như vậy.

Điểm yếu vẫn còn rõ ở shirt/blouse: chỉ có 19 mẫu và model chỉ đoán đúng 9, tức recall 9 trên 19. Với ảnh bị noise hoặc blur mạnh, Macro-F1 tụt xuống khoảng 0,55 và 0,59. Nhóm em giữ nguyên model và ngưỡng để mô tả đúng các lỗi này, không chỉnh lại dựa trên assessment.


## Slide 18. Kiểm chứng bản xuất v1 — Lâm

Ở v1, lượt chạy trên A100 đã đánh giá xong, nhưng bản ONNX bị bộ kiểm tra chặn lại: sai số logit lớn nhất khoảng 1,54 nhân 10 mũ trừ 4, vượt ngưỡng nhóm em đặt là 10 mũ trừ 4. Nhãn trên các ca kiểm tra vẫn trùng, nhưng như vậy chưa đủ để bỏ qua gate.

Nhóm em chỉ sửa phần thiết lập export ONNX, còn checkpoint, temperature và ngưỡng confidence giữ nguyên. Bản export mới đạt cùng tiêu chí trên 64 ca của giao thức kiểm tra ban đầu. Phần xác minh bổ sung chạy trên CPU, runtime ghi riêng, còn kết quả A100 gốc vẫn giữ nguyên. Bản sửa có thời gian chạy khác, nên nhóm em không dùng lại con số tăng tốc của bản export cũ.

Demo nhận vào một crop trang phục thuộc sáu lớp, trả về xác suất, kèm cờ cần kiểm tra khi confidence thấp. Trang web tĩnh đã được chuẩn bị để công bố kết quả, còn API dự đoán chạy riêng. Có ba thứ nhóm em chưa kiểm định: ảnh chụp điện thoại bất kỳ, khả năng phát hiện nhiều vật thể, và vận hành ở mức production. Đó là phạm vi của phiên bản hiện tại.


## Slide 19. Vòng tối ưu có đối chứng — Lâm

Sau v1, nhóm em thấy model vẫn còn overfit, nên làm thêm một vòng tối ưu có đối chứng với tám cấu hình fine-tuning. C0 giữ nguyên recipe gốc. Các probe lần lượt thay learning rate của backbone, weight decay, label smoothing, dropout hoặc augmentation ánh sáng, và có hai cấu hình kết hợp vài yếu tố. Lượt nào cũng chạy 18 epoch từ cùng pretrained weights ImageNet, giữ nguyên architecture và batch size.

Discovery dùng seed 101 và chọn ra hai challenger là C4 và C5. Sau đó nhóm em chạy C0 cùng hai challenger với hai seed mới là 202 và 303. Khác với phần Fashion-MNIST, lần này chỉ điểm của hai seed mới quyết định recipe, không cộng điểm discovery vào. C4, tức tăng dropout từ 0,2 lên 0,4, nhỉnh hơn một chút: selection F1 trung bình khoảng 0,842, so với 0,841 của C0.

Cuối cùng, C0 và C4 đều chạy với ba seed final đã cố định. Tổng cộng 20 lượt, 360 epoch, controller chạy xong trong gần 90 phút. Đây là thời gian tiến trình ghi nhận, không phải GPU kernel time.


## Slide 20. Tuning không chứng minh tăng F1 — Lâm

Nhưng thắng trên selection chưa chắc đã tốt hơn trên assessment. Trên assessment, C0 có F1 trung bình khoảng 0,775, còn C4 khoảng 0,766, tức C4 thấp hơn khoảng 0,008. Khoảng bootstrap ghép cặp theo 221 nhóm ảnh đi từ khoảng âm 0,038 đến dương 0,018. Nghĩa là chưa có bằng chứng C4 tốt hơn.

Dù vậy, nhóm em vẫn giữ model đã chọn theo selection, không đổi model sau khi nhìn assessment. Seed triển khai 314 cũng đã chốt từ trước. Với seed này, C4 đạt accuracy khoảng 82,3% và F1 khoảng 0,776, thấp hơn v1 seed 314 trên cùng tập. Đây là một kết quả âm, và nhóm em báo cáo đầy đủ.

Overfitting cũng vẫn còn: ở epoch cuối, C4 có train accuracy gần 100%, trong khi trên selection chỉ khoảng 87%. Tăng dropout chưa giải quyết hết vấn đề. Thêm nữa, assessment đã được xem từ v1, nên đây chỉ là kiểm tra regression mang tính thăm dò, không phải một blind test mới.


## Slide 21. Calibration tách riêng và đánh đổi coverage — Lâm

Ở v1, cùng một tập calibration vừa dùng để fit temperature vừa dùng để chọn ngưỡng. Sang v2, nhóm em chia calibration theo source/duplicate component thành hai phần: một phần fit temperature, phần còn lại đánh giá ngưỡng. Ở phần thứ hai, mỗi component lấy một đại diện đã chọn từ trước.

Nhóm em xét 101 ngưỡng, dùng cận trên một phía của tỷ lệ lỗi theo binomial và hiệu chỉnh cho cả lưới. Ngưỡng được chọn là 0,88. Trên calibration có 423 đại diện được nhận, trong đó 6 lỗi, cho cận lỗi khoảng 4,4%. Cận này tính theo các đại diện component với trọng số bằng nhau, không phải bảo đảm cho mọi crop. Nó còn phụ thuộc giả định về độc lập và phân phối; calibration cũng đã được xem trong quá trình phát triển trước đó.

Trên assessment cũ, policy mới nhận 182 trên 327 crop, đúng 173: accepted accuracy 95,05%, coverage 55,7%. V1 có coverage gần 70% nhưng accepted accuracy chỉ 92,5%. Hai policy còn khác model và cách calibration, nên đây là đánh đổi quan sát được, không thể quy riêng cho ngưỡng. Phần còn lại được gắn cờ cần review; nhóm em chưa đo kết quả xử lý của con người.

## Slide 22. Dịch vụ PyTorch và giới hạn ONNX — Lâm

Service PyTorch trên CPU đã được kiểm tra trên toàn bộ 327 crop: nhãn và quyết định review khớp với kết quả Colab. Luồng nhận PNG và API contract cũng qua kiểm tra. Trong 60 giây thử HTTP local, service hoàn tất 5.312 request, không có lỗi, p95 khoảng 31 mili giây. Nhưng workload chỉ lặp lại hai ảnh JPEG với vùng crop cố định trên một máy, nên chưa nói được năng lực production.

ONNX qua 64 ca ban đầu nhưng khi mở rộng lên 327 crop thì sai số logit lớn nhất khoảng 1,35 nhân 10 mũ trừ 4, vượt giới hạn 10 mũ trừ 4. Nhãn vẫn trùng, nhưng tiêu chí số không đạt. Ba biến thể export bổ sung cũng chưa qua. Vì vậy nhóm em dùng PyTorch cho demo, còn ONNX ghi là experimental.

Quy trình đã kiểm tra được việc chọn model, calibration và serving. Tuy nhiên, assessment cũ chưa chứng minh tuning giúp tăng F1. Phần tiếp theo bổ sung đối chứng huấn luyện và một cohort RGB ngoài dữ liệu đã dùng theo hồ sơ project.

## Slide 23. Đối chứng recipe và checkpoint — Lâm

Quay lại Fashion-MNIST, đây là vòng đối chứng đã nhắc ở slide 14. Thiết kế được chốt trước: năm family, mỗi family có C0 và recipe OPT3 đã chọn, ba seed mới 1009, 2027, 3037, mỗi lượt 60 epoch. Các cặp giữ cùng split 54.000/6.000, engine, seed khởi tạo và thiết lập DataLoader. Ba checkpoint CE, F1 và epoch 60 lấy từ cùng một lượt train. Vì vậy 30 lượt train tạo ra 90 ô kết quả, không phải 90 thí nghiệm độc lập. Vòng này chạy A100 khoảng 4,3 tiếng.

Tiêu chí chính là CNN selected so với C0 tại epoch 60. F1 trung bình tăng từ 0,9282 lên 0,9291, chênh chưa tới 0,001. Ba seed có chênh lệch lần lượt dương 0,0049, âm 0,0017 và âm 0,0004: chỉ một trong ba cặp tăng. Như vậy chưa có lợi thế nhất quán. Nếu chọn checkpoint theo F1, selected đạt 0,9313 so với 0,9299 của C0, nhưng đó là kết quả phụ trên chính validation dùng để chọn checkpoint.

MLP và Transformer có chênh lệch lớn hơn ở epoch 60. Phép đối chứng này so sánh được cả recipe dưới cùng quy tắc checkpoint, chưa tách riêng tác động của từng hyperparameter. Seed mới cũng không biến validation cũ thành dữ liệu chưa từng được dùng.

## Slide 24. Đánh giá RGB trên cohort khóa trước — Lâm

Với RGB, nhóm em khóa giao thức và sáu checkpoint C0/C4 trước khi suy luận, dùng ba seed 314, 2718 và 5772. Ảnh được chọn theo thứ tự hash cố định từ nhóm giấy phép 3 và 5 của Fashionpedia. Nhóm em loại các source/duplicate component chạm dữ liệu cũ theo hồ sơ tiếp xúc đã ghi. Cohort cuối có 2.766 crop thuộc 1.826 component, đủ sáu class. Suy luận chạy local trên CPU; notebook A100 dùng lại đúng bộ kết quả đó.

C0 có Macro-F1 trung bình khoảng 0,7745, C4 khoảng 0,7815, chênh khoảng 0,0070. Bootstrap ghép cặp theo component, 2.000 lần lấy mẫu, cho khoảng 95% từ âm 0,00036 đến dương 0,01512. Khoảng vẫn chứa 0, nên chưa khẳng định C4 tốt hơn. Trên assessment cũ, chênh lệch mang dấu âm. Nhóm em giữ cả hai để thể hiện sự khác nhau giữa các cohort.

Riêng C4 seed 314, policy đã khóa nhận 1.506 crop và đúng 1.445: accepted accuracy 95,95% tại coverage 54,45%.

Cohort này vẫn cùng nguồn Fashionpedia, có dịch chuyển theo nhóm giấy phép, không phải dữ liệu ngoài nguồn hay một benchmark IID. Việc loại trừ dữ liệu cũ chỉ dựa trên hồ sơ đã ghi, không loại trừ được tiếp xúc khi pretraining hoặc những lần xem ảnh không có trong log. Khoảng bootstrap là cho các model cố định này, còn cận rủi ro từ calibration cũ không tự động áp dụng sang cohort mới.

Tiếp theo, bạn Cầu sẽ phân tích chính sách chấp nhận theo từng lớp.

## Slide 25. Chấp nhận tự động theo lớp — Cầu

Con số 95,95% là tỷ lệ gộp. Vậy từng lớp có được xử lý tốt như nhau không? Ở đây nhóm em chỉ dùng C4 seed 314, temperature khoảng 5,35 và ngưỡng 0,88 đã khóa. Số liệu được tính từ logits và nhãn đã lưu, không đổi model hay ngưỡng.

Bảng trên slide gom theo nhãn thật. Coverage là phần được nhận trong tổng số crop của lớp; accepted accuracy là tỷ lệ đúng trong phần đó. Với shirt/blouse, chỉ 33 trên 182 crop được nhận, coverage khoảng 18%. Trong 33 crop đó có 22 đúng và 11 sai; 149 crop còn lại được gắn cờ review. Pants tốt hơn nhiều: nhận 361 trên 476 crop, đúng 360, tức coverage gần 76% và accepted accuracy 99,7%.

Vì vậy, tỷ lệ gộp 95,95% không có nghĩa mọi lớp đều đạt 95%. Nhãn thật chỉ có khi đánh giá offline. Báo cáo còn có bảng theo nhãn dự đoán, với một mẫu số khác: đầu ra shirt/blouse được nhận có 22 trên 24 đúng, tức 91,67%; đầu ra jacket là 76 trên 88, tức 86,36%. Hai bảng không được dùng thay cho nhau.

Đây là phân tích sau khi xem kết quả. Nhóm em giữ nguyên policy để mô tả điểm yếu; muốn chọn ngưỡng theo từng loại output thì cần calibration phù hợp và một tập đánh giá chưa dùng để chọn ngưỡng.

Em xin chuyển lại cho bạn Lâm để kết luận.

## Slide 26. Kết luận và phạm vi bằng chứng — Lâm

Để kết lại, vòng bổ sung trả lời hai câu hỏi. Một là recipe đã chọn còn giữ lợi thế khi có đối chứng cùng điều kiện không? Với CNN tại tiêu chí chính epoch 60, chưa có cải thiện nhất quán. Hai là chất lượng RGB có giữ được trên cohort ngoài dữ liệu đã dùng theo hồ sơ project không? C4 có F1 cao hơn, nhưng khoảng bootstrap vẫn chứa 0. Kết quả này giúp nhóm em thấy giới hạn của việc chọn model thay vì chỉ báo điểm cao nhất.

Về tái lập, 90 ô kết quả được tính lại từ 540.000 dòng dự đoán. Khi suy luận lại 90 checkpoint trên runtime CPU khác, có 11 nhãn thay đổi trên 540.000 cặp checkpoint–ảnh; chênh lệch NLL trung bình lớn nhất khoảng 1,55 nhân 10 mũ trừ 5. Nhóm em giữ số A100 gốc và ghi riêng khác biệt runtime, không tuyên bố khớp tuyệt đối.

Hạn chế còn lại là validation đã được dùng trong phát triển, RGB vẫn cùng nguồn, số seed ít và service mới kiểm tra local. PyTorch là backend đã qua kiểm tra; ONNX v2 chưa đạt dung sai mở rộng. Hướng tiếp theo là tập ảnh ngoài nguồn với giao thức khóa trước và workload dịch vụ thực tế.

AI hỗ trợ viết LaTeX và phát triển pipeline. Phần trình bày của nhóm em đến đây là kết thúc. Em cảm ơn thầy và các bạn đã lắng nghe ạ.

## Phụ lục số liệu Fashion-MNIST

Kết quả **OPT3 validation final** trước vòng đối chứng bổ sung, checkpoint Macro-F1, training seeds 42/43/44, mỗi seed đánh giá trên 6.000 ảnh. SD là sample SD giữa ba seed; accuracy SD dùng đơn vị điểm phần trăm.

| Family | Recipe | Accuracy mean (%) | Accuracy SD (điểm %) | Macro-F1 mean ± SD |
|---|---|---:|---:|---:|
| Linear | C1 | 86,856 | 0,010 | 0,86851 ± 0,00024 |
| MLP | C7 | 91,439 | 0,167 | 0,91448 ± 0,00153 |
| CNN | C5 | 93,278 | 0,059 | 0,93308 ± 0,00027 |
| GRU | C1 | 91,400 | 0,328 | 0,91367 ± 0,00376 |
| Transformer | C1 | 91,433 | 0,120 | 0,91431 ± 0,00096 |


## Nguồn kết quả

- `output/a1_final/data/publication_data.json` và `output/a1_final/evidence/`: cấu hình, selection, lịch sử, predictions và kết quả nghiên cứu Fashion-MNIST.
- `verification/OPT3_INDEPENDENT_STATISTICS_1d2d83edf1dc.json`: đối chiếu 61 tasks, 3.660 epochs và 338.400 prediction rows.
- `output/garmentlab_improve_return_20261028/FINAL_SUMMARY_V2.json`: kết quả GarmentLab, calibration và chính sách review.
- `verification/GARMENTLAB_IMPROVE_RETURN_AUDIT_20261028.json`: kiểm tra source, metrics, calibration và bootstrap từ artifacts.
- `verification/GARMENTLAB_IMPROVE_LOCAL_PYTORCH_SERVICE_20261028.json`, `GARMENTLAB_IMPROVE_LOCAL_ORIGINAL_ONNX_BATCHES_20261028.json` và `GARMENTLAB_IMPROVE_LOCAL_HTTP_20261028.json`: kiểm tra dịch vụ và phạm vi kết luận kỹ thuật.

- `output/confirmatory_return_20261030/fashion_confirmation/analysis/summary.json` và `verification/CONFIRMATORY_STATISTICS_20261030.json`: 30 quỹ đạo, 90 checkpoint và phép đối chiếu độc lập từ 540.000 dòng dự đoán.
- `docs/a1_extension_research/PROSPECTIVE_RGB_AUDIT_20261029.md` và `assets/prospective_completed/PROSPECTIVE_RESULTS.json`: giao thức, cohort và số liệu RGB đã khóa trước.
- `verification/CONFIRMATORY_CHECKPOINT_CPU_REPLAY_20261030.json`: kiểm tra suy luận khác runtime.
