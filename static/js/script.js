$(document).ready(function () {
  // Global variables
  let currentImage = null;
  let currentAnnotatedImage = null;

  // Upload dataset
  $("#uploadDatasetBtn").click(function () {
    const personName = $("#personName").val().trim();
    if (!personName) {
      updateStatus("Please enter a person name", "error");
      return;
    }

    const files = $("#datasetFiles")[0].files;
    if (files.length === 0) {
      updateStatus("Please select a folder with images", "error");
      return;
    }

    const formData = new FormData();
    formData.append("person_name", personName);
    for (let i = 0; i < files.length; i++) {
      formData.append("dataset", files[i]);
    }

    updateStatus("Uploading dataset...", "processing");
    $("#progressBar").css("width", "0%");

    $.ajax({
      url: "/upload_dataset",
      type: "POST",
      data: formData,
      processData: false,
      contentType: false,
      success: function (response) {
        updateStatus(
          `Uploaded ${response.image_count} images for ${response.person_name}`,
          "success"
        );
        $("#personName").val("");
        $("#datasetFiles").val("");
      },
      error: function (xhr) {
        updateStatus(xhr.responseJSON.error || "Upload failed", "error");
      },
    });
  });

  // Train new model
  $("#trainNewBtn").click(function () {
    if (
      confirm(
        "This will create a new model. Existing model will be replaced. Continue?"
      )
    ) {
      trainModel(false);
    }
  });

  // Continue training
  $("#continueTrainBtn").click(function () {
    trainModel(true);
  });

  // Load existing model
  $("#loadModelBtn").click(function () {
    updateStatus("Loading model...", "processing");
    $("#progressBar").addClass("progress-bar-striped progress-bar-animated");
    $("#progressBar").css("width", "100%");

    $.ajax({
      url: "/load_model",
      type: "POST",
      success: function (response) {
        updateStatus("Model loaded successfully", "success");
        $("#recognizeBtn").prop("disabled", false);
      },
      error: function (xhr) {
        updateStatus(xhr.responseJSON.error || "Failed to load model", "error");
      },
      complete: function () {
        $("#progressBar").removeClass(
          "progress-bar-striped progress-bar-animated"
        );
        $("#progressBar").css("width", "0%");
      },
    });
  });

  // Handle test image selection
  $("#testImage").change(function () {
    const file = this.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = function (e) {
        currentImage = e.target.result; // Simpan Data URL gambar asli
        // Tampilkan gambar asli sebagai preview sebelum recognition
        $("#imageContainer").html(
          `<h5>Original Image:</h5><img src="${currentImage}" class="img-fluid" alt="Test Image">`
        );
        $("#resultsContainer").html(""); // Kosongkan hasil sebelumnya
        updateStatus("Image loaded. Ready for recognition.", "ready");
      };
      reader.readAsDataURL(file);
    }
  });

  // Recognize face
  $("#recognizeBtn").click(function () {
    if (!$("#testImage")[0].files[0]) {
      // Cek apakah file sudah dipilih
      updateStatus("Please select an image first", "error");
      return;
    }

    const file = $("#testImage")[0].files[0];
    const formData = new FormData();
    formData.append("image", file);

    updateStatus("Recognizing faces...", "processing");
    $("#progressBar").css("width", "50%");
    $("#imageContainer").append(
      // Tambahkan pesan bahwa gambar hasil akan muncul
      `<div id="annotatedImagePlaceholder" class="mt-3"><h5>Recognized Image:</h5><p>Processing...</p></div>`
    );

    $.ajax({
      url: "/recognize",
      type: "POST",
      data: formData,
      processData: false,
      contentType: false,
      success: function (response) {
        if (response.success) {
          displayResults(response); // Fungsi ini akan menangani teks hasil

          // Buat path untuk kedua gambar
          const originalImageUrl = `/uploads/${response.original_image}`;
          const annotatedImageUrl = `/uploads/${response.annotated_image}`;

          // Update #imageContainer untuk menampilkan kedua gambar
          // Kita akan menampilkan gambar asli di atas dan gambar hasil anotasi di bawah
          // atau bisa juga menggunakan layout kolom Bootstrap jika diinginkan
          $("#imageContainer").html(`
            <div class="row">
              <div class="col-md-6">
                <h5>Original Image:</h5>
                <img src="${originalImageUrl}" class="img-fluid" alt="Original Image">
              </div>
              <div class="col-md-6">
                <h5>Recognized Image:</h5>
                <img src="${annotatedImageUrl}" class="img-fluid" alt="Annotated Image">
              </div>
            </div>
          `);

          updateStatus(
            `Recognized ${response.faces_count} face(s) in ${response.processing_time}`,
            "success"
          );
        } else {
          // Jika tidak ada wajah terdeteksi atau error dari backend tapi bukan error AJAX
          $("#imageContainer").html(
            // Tampilkan gambar asli saja
            `<h5>Original Image:</h5><img src="/uploads/${
              response.original_image
            }" class="img-fluid" alt="Original Image">
             <p class="mt-2 text-info">${
               response.message ||
               "Recognition completed, no faces detected or other issue."
             }</p>`
          );
          displayResults(response); // Tetap tampilkan pesan dari backend
          updateStatus(response.message || "Recognition completed", "ready");
        }
        $("#progressBar").css("width", "100%");
        setTimeout(() => $("#progressBar").css("width", "0%"), 1000);
      },
      error: function (xhr) {
        updateStatus(xhr.responseJSON.error || "Recognition failed", "error");
        $("#progressBar").css("width", "0%");
        // Jika error, mungkin hanya tampilkan gambar asli yang sudah di-load
        if (currentImage) {
          $("#imageContainer").html(
            `<h5>Original Image:</h5><img src="${currentImage}" class="img-fluid" alt="Test Image">
                 <p class="mt-2 text-danger">Error during recognition.</p>`
          );
        } else {
          $("#imageContainer").html(
            `<p class="mt-2 text-danger">Error during recognition and no image preview available.</p>`
          );
        }
      },
    });
  });

  // Train model helper function
  function trainModel(isUpdate) {
    updateStatus(
      isUpdate ? "Continuing training..." : "Training new model...",
      "processing"
    );
    $("#progressBar").addClass("progress-bar-striped progress-bar-animated");
    $("#progressBar").css("width", "50%");

    $.ajax({
      url: "/train_model",
      type: "POST",
      data: { is_update: isUpdate },
      success: function (response) {
        updateStatus(response.message, "success");
        $("#recognizeBtn").prop("disabled", false);
        $("#progressBar").css("width", "100%");

        // Show training summary
        const summary = `
                    <strong>Training Summary:</strong><br>
                    - Faces trained: ${response.faces_count}<br>
                    - People: ${response.people_count}<br>
                    - Time: ${response.training_time}
                `;
        $("#resultsContainer").html(summary);

        setTimeout(() => $("#progressBar").css("width", "0%"), 1000);
      },
      error: function (xhr) {
        updateStatus(xhr.responseJSON.error || "Training failed", "error");
        $("#progressBar").css("width", "0%");
      },
      complete: function () {
        $("#progressBar").removeClass(
          "progress-bar-striped progress-bar-animated"
        );
      },
    });
  }

  // Display recognition results
  function displayResults(response) {
    let html = "";

    if (response.faces_count === 0 && response.success) {
      // Hanya jika sukses tapi 0 wajah
      html = `<p>${response.message || "No faces detected in the image."}</p>`;
    } else if (response.faces_count > 0) {
      // Hanya jika ada wajah terdeteksi
      html += `<p class="mt-3"><strong>Found ${response.results.length} face(s) with details:</strong></p>`;
      response.results.forEach((result, index) => {
        const statusClass = result.is_recognized
          ? "recognized"
          : "not-recognized";
        const confidenceText = result.is_recognized
          ? ` (${result.confidence} confidence)`
          : "";

        html += `
          <div class="face-result ${statusClass}">
            <strong>Face ${index + 1}:</strong> ${
          result.person
        }${confidenceText}
          </div>
        `;
      });
    }

    if (response.processing_time) {
      html += `<p class="mt-2"><small>Processing time: ${response.processing_time}</small></p>`;
    }
    $("#resultsContainer").html(html);
  }

  // Update status bar
  function updateStatus(message, type) {
    $("#statusText").text(message);

    // Reset classes
    $("#statusText").removeClass("text-success text-danger text-primary");
    $("#progressBar").removeClass("bg-success bg-danger bg-primary");

    switch (type) {
      case "success":
        $("#statusText").addClass("text-success");
        $("#progressBar").addClass("bg-success");
        break;
      case "error":
        $("#statusText").addClass("text-danger");
        $("#progressBar").addClass("bg-danger");
        break;
      case "processing":
        $("#statusText").addClass("text-primary");
        $("#progressBar").addClass("bg-primary");
        break;
      default:
        // 'ready' state - no special classes
        break;
    }
  }
});
